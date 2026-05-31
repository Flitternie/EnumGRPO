#!/usr/bin/env python3
"""scale_databases.py -- Deterministically scale SWAN DuckDB databases.

For each database and each requested scale factor, produces a scaled copy of
the DuckDB file under:

    <out_root>/<scale_label>/<db_id>.duckdb

Two regimes:
  * scale <= 1.0  -- ground-truth-safe subsampling:
      Rows whose values appear in any gold-SQL answer are always kept
      (protected set, computed via SQL).  Only "irrelevant" rows are removed
      to hit the target count.  Point-lookup answers are always preserved.

  * scale >  1.0  -- distribution-preserving synthetic expansion:
      New rows are sampled from the observed per-column distributions with
      fresh synthetic identities (new integer PKs, prefixed text values).
      They never exactly match any gold-answer value, so the agent cannot
      shortcut them with a direct WHERE lookup and cannot dedup them away.
      Aggregation ground truth is preserved in expectation (same distribution).

Usage:
    python scalability/scale_databases.py [OPTIONS]

Options:
    --db_dir DIR        Source DuckDB directory  (default: swan/database)
    --query_file FILE   SWAN query JSONL          (default: swan/evaluation.jsonl)
    --out_root DIR      Output root directory     (default: exp/scaled_dbs)
    --scales FLOATS     Comma-separated scale factors to generate
    --db DB_ID          Only scale this database (repeatable)
    --seed INT          Random seed for determinism (default: 42)
    --force             Overwrite existing scaled databases
    --list              Print the scale plan and exit (dry run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:
    raise SystemExit("duckdb is required: pip install duckdb")

# ---------------------------------------------------------------------------
# Per-database scale configuration
# ---------------------------------------------------------------------------

DB_SCALE_CONFIG: dict[str, dict[str, Any]] = {
    "california_schools": {
        "primary_table": "schools",
        "scales": [0.25, 0.5, 1.0, 2.0, 4.0],
    },
    "european_football_2": {
        "primary_table": "Player_Attributes",
        "scales": [0.25, 0.5, 1.0, 2.0, 4.0],
        "min_rows_tables": ["Country", "League"],
        # FK cols in fact tables that should sample from the reference table's
        # column (not a synthetic range): {fact_col: (ref_table, ref_col)}
        "fk_references": {
            "player_api_id": ("Player", "player_api_id"),
            "player_fifa_api_id": ("Player", "player_fifa_api_id"),
        },
    },
    "formula_1": {
        "primary_table": "races",
        "scales": [0.25, 0.5, 1.0, 2.0, 4.0],
        "min_rows_tables": ["seasons", "status"],
    },
    "superhero": {
        "primary_table": "superhero",
        "scales": [0.25, 0.5, 1.0, 2.0, 4.0],
        "min_rows_tables": [
            "alignment", "attribute", "colour",
            "gender", "publisher", "race", "superpower",
        ],
        # explicit FK aliases: {fact_col: pk_col_in_primary_table}
        # hero_attribute.hero_id and hero_power.hero_id both reference superhero.id
        "fk_aliases": {"hero_id": "id"},
    },
}

DEFAULT_SEED = 42
_SYNTH_PREFIX = "__s_"   # prefix for synthetic text-identity values


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scale_label(scale: float) -> str:
    return f"{scale:.2f}x".replace(".", "_")


def _table_info(conn: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [{"name": r[1], "type": str(r[2] or ""), "pk": bool(r[5])} for r in rows]


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def _int_pk_columns(cols: list[dict[str, Any]]) -> list[str]:
    out = []
    for c in cols:
        if not c["pk"]:
            continue
        t = c["type"].upper()
        if any(x in t for x in ("INT", "BIGINT", "SMALLINT", "HUGEINT", "SERIAL")):
            out.append(c["name"])
    return out


def _hash_seed(table: str, seed: int) -> int:
    raw = hashlib.sha256(f"{table}:{seed}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def _all_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in conn.execute("SHOW TABLES").fetchall()
            if not r[0].startswith("sqlite_")]


def _is_numeric_type(type_str: str) -> bool:
    t = type_str.upper()
    return any(x in t for x in ("INT", "BIGINT", "SMALLINT", "HUGEINT",
                                  "FLOAT", "DOUBLE", "REAL", "DECIMAL",
                                  "NUMERIC", "SERIAL"))


def _is_text_type(type_str: str) -> bool:
    t = type_str.upper()
    return any(x in t for x in ("VARCHAR", "TEXT", "CHAR", "STRING", "BLOB"))


def _row_hash_sql(col_names: list[str]) -> str:
    """SQL expression producing a stable md5 hash for a full row."""
    concat = " || '|' || ".join(
        f"COALESCE(CAST(\"{c}\" AS VARCHAR), '')" for c in col_names
    )
    return f"md5({concat})"


# ---------------------------------------------------------------------------
# Step 1: Build protected-row set  (pure SQL, no full-table Python scan)
# ---------------------------------------------------------------------------

def build_protected_info(
    src: Path,
    db_id: str,
    queries: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], set[str]]:
    """Return (protected_hashes, answer_values) for the PRIMARY TABLE only.

    protected_hashes: {primary_table -> set of md5 row-hashes that must not be removed}
    answer_values:    all string values from gold-SQL answers (used to avoid
                      synthetic rows accidentally matching gold answers)

    Only the primary table is subsampled (to preserve JOIN integrity and gold
    truth for other tables).  We protect primary-table rows whose text columns
    match any WHERE-clause literal or answer value from the gold SQL.
    """
    db_queries = [q for q in queries if q.get("db") == db_id]
    if not db_queries:
        return {}, set()

    conn = duckdb.connect(str(src), read_only=True)
    try:
        tables = _all_tables(conn)
        # Determine which table is the primary table for this DB.
        primary_table = DB_SCALE_CONFIG.get(db_id, {}).get("primary_table", "")

        # Collect answer values and WHERE-clause literals.
        answer_values: set[str] = set()
        text_answer_values: set[str] = set()

        for q in db_queries:
            sql = (q.get("sql") or "").strip()

            # Always collect pre-stored answer values from the JSONL (reliable
            # even when gold SQL cannot execute on the schema-corrupted DB).
            for row in q.get("answer") or []:
                for v in (row if isinstance(row, (list, tuple)) else [row]):
                    if v is not None:
                        sv = str(v).strip()
                        answer_values.add(sv)
                        # Only protect on text values that are specific enough:
                        # skip numerics, short strings (≤2 chars like 'CA', 'CA'),
                        # and pure state/country codes that match every row.
                        if (sv
                                and not sv.lstrip("-").replace(".", "", 1).isdigit()
                                and len(sv) > 2):
                            text_answer_values.add(sv)

            if not sql:
                continue
            # Additionally try to execute the SQL for cases where the answer
            # field may be incomplete or where the ref DB has the full schema.
            try:
                for row in conn.execute(sql).fetchall():
                    for v in row:
                        if v is not None:
                            sv = str(v).strip()
                            answer_values.add(sv)
                            if (sv
                                    and not sv.lstrip("-").replace(".", "", 1).isdigit()
                                    and len(sv) > 2):
                                text_answer_values.add(sv)
            except Exception:
                pass

        # Use only answer values (actual query results) for row protection.
        # SQL WHERE-clause literals (e.g. StatusType = 'Active') are filter
        # conditions that may match nearly every row -- including them causes
        # mass over-protection that defeats downscaling entirely.
        all_protect_values = text_answer_values
        protected: dict[str, set[str]] = {}

        # Pass 1: text-value matching scan.
        if all_protect_values:
            conn.execute("CREATE OR REPLACE TEMP TABLE _av (v VARCHAR)")
            conn.executemany("INSERT INTO _av VALUES (?)", [[v] for v in all_protect_values])

            for t in tables:
                cols = _table_info(conn, t)
                col_names = [c["name"] for c in cols]
                # For all tables: scan text columns.
                # For the primary table also scan integer columns (catches
                # integer-literal WHERE clauses like raceId = 901).
                text_cols = [c["name"] for c in cols if _is_text_type(c["type"])]
                int_cols   = [c["name"] for c in cols if _is_numeric_type(c["type"])]
                if t == primary_table:
                    check_cols = text_cols + int_cols
                else:
                    check_cols = text_cols
                if not check_cols:
                    continue
                # Quick sample to skip tables with no overlap -- but always
                # scan the primary table fully since that's what we subsample.
                if t != primary_table:
                    sample_cols = check_cols[:8]
                    sample_vals: set[str] = set()
                    for tc in sample_cols:
                        rows_s = conn.execute(
                            f'SELECT DISTINCT CAST("{tc}" AS VARCHAR) FROM "{t}" '
                            f'WHERE "{tc}" IS NOT NULL LIMIT 200'
                        ).fetchall()
                        sample_vals.update(r[0] for r in rows_s if r[0])
                        if sample_vals.intersection(all_protect_values):
                            break
                    if not sample_vals.intersection(all_protect_values):
                        continue
                where_clause = " OR ".join(
                    f'CAST("{c}" AS VARCHAR) IN (SELECT v FROM _av)'
                    for c in check_cols
                )
                hash_expr = _row_hash_sql(col_names)
                try:
                    hashes = {r[0] for r in conn.execute(
                        f'SELECT {hash_expr} FROM "{t}" WHERE {where_clause}'
                    ).fetchall()}
                except Exception:
                    hashes = set()
                if hashes:
                    protected[t] = hashes

            conn.execute("DROP TABLE IF EXISTS _av")

        return protected, answer_values
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 2: Ground-truth-safe subsampling  (scale <= 1.0)
# ---------------------------------------------------------------------------

def _subsample_table_safe(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    target_rows: int,
    seed: int,
    protected_hashes: set[str],
) -> None:
    """Replace *table* content with a protected-first subsample.

    Protected rows (whose md5 hash is in protected_hashes) are always kept.
    The remaining quota is filled deterministically from free rows.
    All done in SQL -- no full-table Python iteration.
    """
    cols = _table_info(conn, table)
    col_names = [c["name"] for c in cols]
    hash_expr = _row_hash_sql(col_names)
    ts = _hash_seed(table, seed)

    if protected_hashes:
        # Load protected hashes into a temp table for SQL join.
        conn.execute("CREATE OR REPLACE TEMP TABLE _ph (h VARCHAR)")
        conn.executemany("INSERT INTO _ph VALUES (?)", [[h] for h in protected_hashes])
        n_protected = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE {hash_expr} IN (SELECT h FROM _ph)'
        ).fetchone()[0]
        n_free_needed = max(0, target_rows - n_protected)

        conn.execute(f"""
            CREATE OR REPLACE TABLE "{table}" AS
            SELECT * FROM "{table}"
            WHERE {hash_expr} IN (SELECT h FROM _ph)
            UNION ALL
            SELECT * FROM (
                SELECT * FROM "{table}"
                WHERE {hash_expr} NOT IN (SELECT h FROM _ph)
                ORDER BY md5({hash_expr} || ':{ts}')
                LIMIT {n_free_needed}
            )
        """)
        conn.execute("DROP TABLE IF EXISTS _ph")
    else:
        # No protected rows -- plain deterministic subsample.
        conn.execute(f"""
            CREATE OR REPLACE TABLE "{table}" AS
            SELECT * FROM "{table}"
            ORDER BY md5({hash_expr} || ':{ts}')
            LIMIT {target_rows}
        """)


# ---------------------------------------------------------------------------
# Step 3: Distribution-preserving synthetic expansion  (scale > 1.0)
# ---------------------------------------------------------------------------

def _collect_all_distributions(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    cols: list[dict[str, Any]],
    sample_size: int = 10000,
) -> dict[str, dict[str, Any]]:
    """Sample the table once and build per-column distributions from that sample."""
    col_names = [c["name"] for c in cols]
    if not col_names:
        return {}
    select_expr = ", ".join(f'"{c}"' for c in col_names)
    rows = conn.execute(
        f'SELECT {select_expr} FROM "{table}" USING SAMPLE {sample_size} ROWS'
    ).fetchall()

    if not rows:
        return {c["name"]: {"kind": "const", "value": None} for c in cols}

    from collections import Counter
    dists: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(cols):
        cn, ct = c["name"], c["type"]
        raw = [row[i] for row in rows if row[i] is not None]
        if not raw:
            dists[cn] = {"kind": "const", "value": None}
        elif _is_numeric_type(ct):
            dists[cn] = {"kind": "numeric_sample", "values": [float(v) for v in raw]}
        else:
            counts = Counter(str(v) for v in raw)
            total = sum(counts.values())
            keys = list(counts.keys())
            dists[cn] = {
                "kind": "categorical",
                "values": keys,
                "weights": [counts[k] / total for k in keys],
            }
    return dists


def _sample_value(dist: dict[str, Any], rng: Any, answer_values: set[str]) -> Any:
    """Sample one value (kept for potential future use; main path uses NumPy)."""
    if dist["kind"] == "const":
        return dist["value"]
    if dist["kind"] == "numeric_sample":
        import random
        return random.choice(dist["values"])
    vals, weights = dist["values"], dist["weights"]
    import random
    for _ in range(10):
        v = random.choices(vals, weights=weights, k=1)[0]
        if v not in answer_values:
            return v
    return None


def _expand_table_synthetic(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    n_new: int,
    seed: int,
    answer_values: set[str],
    fk_new_range: dict[str, tuple[int, int, bool]] | None = None,
    fk_sample_from: dict[str, tuple[str, str]] | None = None,
) -> None:
    """Append *n_new* synthetic rows drawn from per-column distributions.

    Uses fully vectorised numpy sampling + pandas DataFrame bulk insert for
    speed.  For a 148k-row table this takes ~3s vs ~2 min with a Python loop.

    Args:
        fk_new_range:     {col: (lo, hi, is_varchar)} -- restrict FK to new
                          synthetic integer range (prevents polluting real rows).
        fk_sample_from:   {col: (ref_table, ref_col)} -- sample FK values from
                          the current full contents of a reference column.
                          Used when the fact table references an entity table
                          that is also being expanded (e.g. Player_Attributes →
                          Player), so synthetic fact rows reference only the
                          new synthetic entity IDs.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        raise SystemExit("numpy and pandas are required: pip install numpy pandas")

    fk_new_range = fk_new_range or {}

    cols = _table_info(conn, table)
    col_names = [c["name"] for c in cols]

    # Find the table's own identity column (auto-increment PK substitute).
    # DuckDB SWAN files have no PK constraints, so use the heuristic: first
    # integer column ending in "id" (or exactly named "id") that is NOT in
    # fk_new_range (those are FK columns, not the table's own PK).
    # Exception: if it would be the ONLY column in the table, it is a FK reference
    # column (e.g. hero_power.power_id after hero_id was dropped), not an
    # auto-increment PK -- leave it as a distribution column.
    own_pk_col: str | None = None
    for c in cols:
        if c["type"].upper() in ("BIGINT", "INTEGER", "INT") and \
                c["name"].lower().endswith("id") and \
                c["name"] not in fk_new_range:
            candidate = c["name"]
            other_cols = [x for x in cols if x["name"] != candidate and x["name"] not in fk_new_range]
            if other_cols:  # only promote to PK if there are other columns beside it
                own_pk_col = candidate
            break

    max_pk = 0
    if own_pk_col:
        max_pk = int(conn.execute(
            f'SELECT MAX(CAST("{own_pk_col}" AS BIGINT)) FROM "{table}"'
        ).fetchone()[0] or 0)

    non_pk_cols = [c for c in cols if c["name"] != own_pk_col]
    dists = _collect_all_distributions(conn, table, non_pk_cols)

    rng_np = np.random.default_rng(seed=_hash_seed(table, seed) % (2**32))

    data: dict[str, Any] = {}

    for c in cols:
        cn, ct = c["name"], c["type"]

        if cn == own_pk_col:
            data[cn] = (np.arange(1, n_new + 1, dtype=np.int64) + max_pk).tolist()
            continue

        # FK column restricted to the synthetic primary key range.
        if cn in fk_new_range:
            lo, hi, is_varchar = fk_new_range[cn]
            int_vals = rng_np.integers(lo, hi + 1, size=n_new).tolist()
            data[cn] = [str(v) for v in int_vals] if is_varchar else int_vals
            continue

        # FK column sampled from a reference table (already expanded in this run).
        fk_sample_from = fk_sample_from or {}
        if cn in fk_sample_from:
            ref_tbl, ref_col = fk_sample_from[cn]
            ref_vals = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{ref_col}" FROM "{ref_tbl}" WHERE "{ref_col}" IS NOT NULL'
            ).fetchall()]
            if ref_vals:
                idxs = rng_np.integers(0, len(ref_vals), size=n_new)
                data[cn] = [ref_vals[i] for i in idxs]
                continue

        dist = dists[cn]

        if dist["kind"] == "const":
            data[cn] = [dist["value"]] * n_new
        elif dist["kind"] == "numeric_sample":
            # Vectorised: sample indices then gather.
            src = np.array(dist["values"], dtype=np.float64)
            idxs = rng_np.integers(0, len(src), size=n_new)
            data[cn] = src[idxs].tolist()
        else:
            # Categorical: numpy multinomial sampling.
            vals = dist["values"]
            weights = np.array(dist["weights"], dtype=np.float64)
            weights /= weights.sum()
            idxs = rng_np.choice(len(vals), size=n_new, p=weights)
            col_vals = [vals[i] for i in idxs]

            # Prefix text-identity columns so they never match gold answers.
            # Skip columns whose values are all numeric strings (e.g. speed
            # stored as VARCHAR), date-like, URL, or FK/code columns.
            looks_numeric = vals and all(
                v.lstrip("-").replace(".", "", 1).replace("e", "", 1).isdigit()
                for v in vals[:20] if v
            )
            if (
                _is_text_type(ct)
                and not looks_numeric
                and not cn.lower().endswith("_id")
                and "code" not in cn.lower()
                and "url"  not in cn.lower()
                and "date" not in cn.lower()
                and "time" not in cn.lower()
                and "birth" not in cn.lower()
                and "speed" not in cn.lower()
                and "rating" not in cn.lower()
            ):
                seq = np.arange(1, n_new + 1, dtype=np.int32)
                col_vals = [
                    f"{_SYNTH_PREFIX}{i}_{v}" if v not in answer_values else f"{_SYNTH_PREFIX}{i}"
                    for i, v in zip(seq.tolist(), col_vals)
                ]
            data[cn] = col_vals

    df = pd.DataFrame(data, columns=col_names)
    conn.execute(f'INSERT INTO "{table}" SELECT * FROM df')


# ---------------------------------------------------------------------------
# Main scale_database function
# ---------------------------------------------------------------------------

def scale_database(
    *,
    src: Path,
    dst: Path,
    primary_table: str,
    scale: float,
    seed: int,
    force: bool,
    min_rows_tables: list[str] | None = None,
    fk_aliases: dict[str, str] | None = None,
    fk_references: dict[str, tuple[str, str]] | None = None,
    protected_hashes: dict[str, set[str]] | None = None,
    answer_values: set[str] | None = None,
) -> None:
    if dst.exists() and not force:
        print(f"  [skip] {dst.name} already exists (use --force to overwrite)", flush=True)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(0o644)  # ensure writable even if source is read-only
    protected_names: set[str] = {t.lower() for t in (min_rows_tables or [])}
    protected_hashes = protected_hashes or {}
    answer_values = answer_values or set()

    conn = duckdb.connect(str(dst))
    try:
        tables = _all_tables(conn)
        if primary_table not in tables:
            raise ValueError(
                f"Primary table '{primary_table}' not found in {src.name}. "
                f"Available: {tables}"
            )

        orig_primary = _row_count(conn, primary_table)
        target_primary = max(1, round(orig_primary * scale))

        if scale <= 1.0:
            # Subsampling regime: only subsample the PRIMARY table.
            # All other tables are kept intact to preserve JOIN integrity and
            # ensure that gold-SQL answers remain valid for non-primary tables.
            # The primary table is the semantically meaningful scaling knob.
            t = primary_table
            orig = _row_count(conn, t)
            target = max(1, round(orig * scale))
            if target < orig:
                prot = protected_hashes.get(t, set())
                _subsample_table_safe(conn, t, target, seed, prot)
                new = _row_count(conn, t)
                print(f"    {t} (primary): {orig:,} -> {new:,} rows  (protected≥{min(len(prot), new):,})", flush=True)
            for t in tables:
                if t == primary_table:
                    continue
                if t.lower() in protected_names:
                    print(f"    {t}: protected reference table (kept intact)", flush=True)
                else:
                    print(f"    {t}: kept intact (JOIN integrity)", flush=True)
        else:
            # Build a shared FK restriction map across all expanded tables.
            #
            # For any column name ending in "id" that appears in more than one
            # expanded table (and stores integer or numeric-string values), we
            # compute the global max across all those tables and assign a single
            # synthetic range [max+1 .. max+n_new_primary].  This ensures that
            # both sides of every cross-table join use matching new IDs.
            #
            # The primary table's id col(s) are treated as the canonical reference
            # for sizing the range; non-primary tables get the same range applied.
            #
            # is_varchar=True → column stores IDs as numeric strings (schema-
            # corrupted DBs); sampled integers are cast to str before insert.
            expanded_tables = [
                t for t in tables
                if t.lower() not in protected_names
            ]

            # Step 1: collect all *id columns per table, noting type.
            table_id_cols: dict[str, dict[str, bool]] = {}  # t → {col: is_varchar}
            for t in expanded_tables:
                t_cols = _table_info(conn, t)
                id_map: dict[str, bool] = {}
                for c in t_cols:
                    if not c["name"].lower().endswith("id"):
                        continue
                    ct = c["type"].upper()
                    if ct in ("BIGINT", "INTEGER", "INT"):
                        id_map[c["name"]] = False
                    elif _is_text_type(ct):
                        sample = conn.execute(
                            f'SELECT "{c["name"]}" FROM "{t}" '
                            f'WHERE "{c["name"]}" IS NOT NULL LIMIT 20'
                        ).fetchall()
                        vals = [r[0] for r in sample if r[0] is not None]
                        if vals and all(
                            v.lstrip("-").replace(".", "", 1).isdigit()
                            for v in vals
                        ):
                            id_map[c["name"]] = True
                table_id_cols[t] = id_map

            # Step 2: find columns shared between 2+ expanded tables.
            from collections import Counter as _Counter
            col_freq = _Counter(
                col
                for id_map in table_id_cols.values()
                for col in id_map
            )
            shared_cols = {col for col, cnt in col_freq.items() if cnt >= 2}

            # Step 3: for each shared col compute global max and synthetic range.
            n_new_primary = target_primary - _row_count(conn, primary_table)
            global_max: dict[str, int] = {}
            for col in shared_cols:
                col_max = 0
                for t, id_map in table_id_cols.items():
                    if col not in id_map:
                        continue
                    val = conn.execute(
                        f'SELECT MAX(CAST("{col}" AS BIGINT)) FROM "{t}"'
                    ).fetchone()[0]
                    if val is not None:
                        col_max = max(col_max, int(val))
                global_max[col] = col_max

            # Use synthetic range sized to the primary table's expansion.
            synth_pk_lo = (max(global_max.values()) + 1) if global_max else 1
            synth_pk_hi = synth_pk_lo + max(1, n_new_primary) - 1

            # Determine is_varchar from any table that has the column.
            fk_range_map: dict[str, tuple[int, int, bool]] = {}
            for col in shared_cols:
                is_var = next(
                    id_map[col]
                    for id_map in table_id_cols.values()
                    if col in id_map
                )
                fk_range_map[col] = (synth_pk_lo, synth_pk_hi, is_var)

            # Extend with explicit FK aliases from DB_SCALE_CONFIG.
            # fk_aliases maps fact-col-name → pk-col-name-in-primary-table.
            # We inherit the synthetic range from the pk col if it was detected;
            # otherwise fall back to the global synth_pk_lo/hi with is_varchar
            # inherited from any detected id col (or default to non-varchar).
            _aliases = fk_aliases or {}
            for fact_col, pk_col in _aliases.items():
                if fact_col in fk_range_map:
                    continue  # already covered by name-match heuristic
                if pk_col in fk_range_map:
                    fk_range_map[fact_col] = fk_range_map[pk_col]
                else:
                    # pk_col not detected by name-match (e.g. named "id" but not
                    # shared across tables). Compute its actual max from the
                    # primary table and derive a correct synthetic range.
                    pk_max = 0
                    try:
                        pk_max = int(conn.execute(
                            f'SELECT MAX(CAST("{pk_col}" AS BIGINT)) '
                            f'FROM "{primary_table}"'
                        ).fetchone()[0] or 0)
                    except Exception:
                        pass
                    alias_lo = pk_max + 1
                    alias_hi = pk_max + max(1, n_new_primary)
                    # Also re-anchor synth_pk_lo/hi if global_max was empty.
                    if not global_max:
                        synth_pk_lo, synth_pk_hi = alias_lo, alias_hi
                    # Determine is_varchar by checking any expanded table for fact_col.
                    is_var_alias = False
                    for t in expanded_tables:
                        t_info = _table_info(conn, t)
                        for c in t_info:
                            if c["name"] == fact_col:
                                is_var_alias = _is_text_type(c["type"].upper())
                                break
                    fk_range_map[fact_col] = (alias_lo, alias_hi, is_var_alias)
                    # Note: pk_col (e.g. superhero.id) is intentionally NOT added
                    # to fk_range_map so it uses the sequential auto-increment PK
                    # path in _expand_table_synthetic.

            if fk_range_map:
                print(
                    f"  Shared id cols: {sorted(fk_range_map)} "
                    f"(synthetic range={synth_pk_lo}..{synth_pk_hi})",
                    flush=True,
                )

            for t in tables:
                if t.lower() in protected_names:
                    print(f"    {t}: protected reference table (kept intact)", flush=True)
                    continue
                orig = _row_count(conn, t)
                n_new = (target_primary - orig) if t == primary_table else max(0, round(orig * scale) - orig)
                if n_new <= 0:
                    continue

                # All expanded tables: restrict shared id columns to the new
                # synthetic range so both sides of every cross-table join use
                # matching new IDs and never reference real existing entities.
                t_col_names = {c["name"] for c in _table_info(conn, t)}
                t_fk_range: dict[str, tuple[int, int, bool]] = {
                    col: rng for col, rng in fk_range_map.items()
                    if col in t_col_names
                }

                # For FK columns with explicit reference tables (fk_references),
                # sample from the already-expanded reference table's column.
                # This handles M:1 relationships where the FK target is another
                # expanded table (e.g. Player_Attributes.player_api_id → Player).
                _fk_refs = fk_references or {}
                t_fk_sample: dict[str, tuple[str, str]] = {
                    col: ref for col, ref in _fk_refs.items()
                    if col in t_col_names
                }
                # Remove fk_ref cols from t_fk_range to avoid double-handling.
                for col in t_fk_sample:
                    t_fk_range.pop(col, None)

                _expand_table_synthetic(
                    conn, t, n_new, seed, answer_values, t_fk_range, t_fk_sample
                )
                new = _row_count(conn, t)
                fk_note = f"  (FK→new {list(t_fk_range.keys())})" if t_fk_range else ""
                ref_note = f"  (FK→ref {list(t_fk_sample.keys())})" if t_fk_sample else ""
                print(f"    {t}: {orig:,} -> {new:,} rows  (+{n_new:,} synthetic{fk_note}{ref_note})", flush=True)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_db_dir   = repo_root / "swan" / "database"
    default_ref_dir  = repo_root / "SWAN" / "databases" / "duckdb"
    default_qfile    = repo_root / "swan" / "evaluation.jsonl"
    default_out_root = repo_root / "exp" / "scaled_dbs"

    p = argparse.ArgumentParser(
        description="Deterministically scale SWAN DuckDB files for scalability experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db_dir",     default=str(default_db_dir))
    p.add_argument("--ref_db_dir", default=str(default_ref_dir),
                   help="Original (uncorrupted) DB dir used for executing gold SQL to "
                        "build protected-row sets. Defaults to SWAN/databases/duckdb.")
    p.add_argument("--query_file", default=str(default_qfile),
                   help="SWAN query JSONL used to build the protected-row set")
    p.add_argument("--out_root",   default=str(default_out_root))
    p.add_argument("--scales",     help="Comma-separated scale factors")
    p.add_argument("--db", action="append", default=[],
                   help="Only scale this db_id (repeatable)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true", help="Dry run")
    args = p.parse_args()

    db_dir     = Path(args.db_dir).resolve()
    ref_db_dir = Path(args.ref_db_dir).resolve()
    out_root   = Path(args.out_root).resolve()
    query_file = Path(args.query_file).resolve()

    queries: list[dict[str, Any]] = []
    if query_file.is_file():
        with query_file.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
    else:
        print(f"Warning: query file not found: {query_file}", file=sys.stderr)

    target_dbs = [d.strip() for d in args.db if d.strip()] or sorted(DB_SCALE_CONFIG.keys())

    global_scales: list[float] | None = None
    if args.scales:
        try:
            global_scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
        except ValueError as e:
            print(f"Error parsing --scales: {e}", file=sys.stderr)
            return 1

    plan: list[dict[str, Any]] = []
    for db_id in target_dbs:
        if db_id not in DB_SCALE_CONFIG:
            print(f"Warning: '{db_id}' not in DB_SCALE_CONFIG, skipping.", file=sys.stderr)
            continue
        cfg = DB_SCALE_CONFIG[db_id]
        scales = global_scales if global_scales is not None else cfg["scales"]
        src = db_dir / f"{db_id}.duckdb"
        if not src.is_file():
            print(f"Warning: source not found: {src}", file=sys.stderr)
            continue
        for scale in scales:
            label = _scale_label(scale)
            plan.append({
                "db_id": db_id, "scale": scale, "label": label,
                "primary_table": cfg["primary_table"],
                "min_rows_tables": cfg.get("min_rows_tables"),
                "fk_aliases": cfg.get("fk_aliases", {}),
                "fk_references": cfg.get("fk_references", {}),
                "src": src,
                "dst": out_root / label / f"{db_id}.duckdb",
            })

    if args.list or not plan:
        print(f"{'DB':<30} {'Scale':>8}  {'Label':<10}  {'Primary table':<25}  Destination")
        print("-" * 110)
        for e in plan:
            print(f"{e['db_id']:<30} {e['scale']:>8.2f}  {e['label']:<10}  "
                  f"{e['primary_table']:<25}  {e['dst']}")
        if not plan:
            print("(no databases matched)")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    # Pre-compute protected info once per db_id.
    protected_cache:     dict[str, dict[str, set[str]]] = {}
    answer_values_cache: dict[str, set[str]]            = {}

    for db_id in dict.fromkeys(e["db_id"] for e in plan):
        src = db_dir / f"{db_id}.duckdb"
        ref_src = ref_db_dir / f"{db_id}.duckdb"
        if not ref_src.is_file():
            ref_src = src  # fall back to corrupted DB if ref not available
            print(f"  Warning: ref DB not found at {ref_src}, using corrupted source for protection.", file=sys.stderr)
        print(f"\nBuilding protected set for '{db_id}' (ref: {ref_src.parent.name})...", flush=True)
        ph, av = build_protected_info(ref_src, db_id, queries)
        protected_cache[db_id]     = ph
        answer_values_cache[db_id] = av
        sizes = {t: len(h) for t, h in ph.items()}
        print(f"  Protected rows per table : {sizes}", flush=True)
        print(f"  Unique answer values     : {len(av)}", flush=True)

    for entry in plan:
        db_id = entry["db_id"]
        print(f"\n[{db_id}] scale={entry['scale']:.2f}x ({entry['label']}) -> {entry['dst']}", flush=True)
        try:
            scale_database(
                src=entry["src"], dst=entry["dst"],
                primary_table=entry["primary_table"],
                scale=entry["scale"], seed=args.seed, force=args.force,
                min_rows_tables=entry["min_rows_tables"],
                fk_aliases=entry.get("fk_aliases", {}),
                fk_references=entry.get("fk_references", {}),
                protected_hashes=protected_cache.get(db_id, {}),
                answer_values=answer_values_cache.get(db_id, set()),
            )
            status = "ok"
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            status = f"error: {exc}"

        manifest.append({
            "db_id": db_id, "scale": entry["scale"],
            "label": entry["label"], "primary_table": entry["primary_table"],
            "db_dir": str(entry["dst"].parent), "status": status,
        })

    manifest_path = out_root / "scale_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\nManifest written to: {manifest_path}", flush=True)

    errors = sum(1 for e in manifest if not e["status"].startswith("ok"))
    print(f"\nDone. {len(manifest) - errors}/{len(manifest)} scaled databases created successfully.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
