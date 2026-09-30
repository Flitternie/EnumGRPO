#!/usr/bin/env python3
"""Prepare a deterministic Spider 1.0 pure-SQL evaluation subset for EnumGRPO.

The gold SQL is used only to construct reference answer tables and difficulty
strata. It is never included in the agent message at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
HARDNESS_ORDER = ("easy", "medium", "hard", "extra")
WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _duckdb_type(
    declared: str,
    sqlite_types: set[str],
    spider_type: str | None = None,
) -> str:
    official = str(spider_type or "").casefold()
    if official == "number":
        # DOUBLE safely represents both integer and fractional values and also
        # handles numeric strings after explicit null-token normalization.
        return "DOUBLE"
    if official == "boolean":
        return "BOOLEAN"
    if official in {"text", "time"}:
        # Keep dates/times as text, matching the SWAN DuckDB assets.  This
        # avoids silently introducing timezone or parser-dependent behavior.
        return "VARCHAR"

    kind = (declared or "").upper()
    non_null_types = sqlite_types - {"null"}
    if "INT" in kind:
        return "BIGINT"
    if any(token in kind for token in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "DOUBLE"
    if "BLOB" in kind:
        return "BLOB"
    if non_null_types and non_null_types <= {"integer"}:
        return "BIGINT"
    if non_null_types and non_null_types <= {"integer", "real"}:
        return "DOUBLE"
    if non_null_types and non_null_types <= {"blob"}:
        return "BLOB"
    return "VARCHAR"


def _extract_dev_assets(archive: Path, extracted_root: Path) -> Path:
    root = extracted_root / "spider_data"
    marker = root / ".enumgrpo-dev-extracted"
    if marker.is_file():
        return root

    with zipfile.ZipFile(archive) as zf:
        dev = json.loads(zf.read("spider_data/dev.json"))
        db_ids = {str(row["db_id"]) for row in dev}
        wanted_prefixes = tuple(f"spider_data/database/{db}/" for db in db_ids)
        wanted_files = {"spider_data/dev.json", "spider_data/tables.json"}
        for member in zf.infolist():
            name = member.filename
            if name in wanted_files or name.startswith(wanted_prefixes):
                target = (extracted_root / name).resolve()
                if extracted_root.resolve() not in target.parents and target != extracted_root.resolve():
                    raise ValueError(f"Unsafe archive member: {name}")
                zf.extract(member, extracted_root)
    marker.touch()
    return root


def _nested_sql(sql: dict[str, Any]) -> list[dict[str, Any]]:
    nested = []
    conditions = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    for condition in conditions:
        if isinstance(condition[3], dict):
            nested.append(condition[3])
        if isinstance(condition[4], dict):
            nested.append(condition[4])
    for key in ("intersect", "except", "union"):
        if sql[key] is not None:
            nested.append(sql[key])
    return nested


def _count_aggregates(units: Iterable[Any]) -> int:
    return sum(1 for unit in units if unit[0] != 0)


def _spider_hardness(sql: dict[str, Any]) -> str:
    component1 = 0
    component1 += bool(sql["where"])
    component1 += bool(sql["groupBy"])
    component1 += bool(sql["orderBy"])
    component1 += sql["limit"] is not None
    component1 += max(0, len(sql["from"]["table_units"]) - 1)
    connectors = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    component1 += sum(token == "or" for token in connectors)
    conditions = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    component1 += sum(condition[1] == WHERE_OPS.index("like") for condition in conditions)

    component2 = len(_nested_sql(sql))
    aggregate_count = _count_aggregates(sql["select"][1])
    aggregate_count += _count_aggregates(sql["where"][::2])
    aggregate_count += _count_aggregates(sql["groupBy"])
    if sql["orderBy"]:
        aggregate_count += _count_aggregates(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    aggregate_count += _count_aggregates(sql["having"])
    others = int(aggregate_count > 1)
    others += int(len(sql["select"][1]) > 1)
    others += int(len(sql["where"]) > 1)
    others += int(len(sql["groupBy"]) > 1)

    if component1 <= 1 and others == 0 and component2 == 0:
        return "easy"
    if ((others <= 2 and component1 <= 1 and component2 == 0)
            or (component1 <= 2 and others < 2 and component2 == 0)):
        return "medium"
    if ((others > 2 and component1 <= 2 and component2 == 0)
            or (2 < component1 <= 3 and others <= 2 and component2 == 0)
            or (component1 <= 1 and others == 0 and component2 <= 1)):
        return "hard"
    return "extra"


def _difficulty_quotas(counts: Counter[str], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {h: sample_size * counts[h] / total for h in HARDNESS_ORDER}
    quotas = {h: int(raw[h]) for h in HARDNESS_ORDER}
    remaining = sample_size - sum(quotas.values())
    for h in sorted(HARDNESS_ORDER, key=lambda x: raw[x] - quotas[x], reverse=True)[:remaining]:
        quotas[h] += 1
    return quotas


def _select_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size > len(rows):
        raise ValueError(f"sample_size must be within 1..{len(rows)}")

    counts = Counter(str(row["complexity"]) for row in rows)
    quotas = _difficulty_quotas(counts, sample_size)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["complexity"])][str(row["db_id"])].append(row)

    rng = random.Random(seed)
    for by_db in grouped.values():
        for bucket in by_db.values():
            rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    # Guarantee coverage of every dev database first.
    for db in sorted({str(row["db_id"]) for row in rows}):
        candidates = [
            row for row in rows
            if row["db_id"] == db and row["_index"] not in selected_ids
        ]
        candidates.sort(
            key=lambda row: (
                -max(0, quotas[str(row["complexity"])]),
                HARDNESS_ORDER.index(str(row["complexity"])),
                rng.random(),
            )
        )
        chosen = candidates[0]
        selected.append(chosen)
        selected_ids.add(int(chosen["_index"]))
        quotas[str(chosen["complexity"])] -= 1

    # Fill the remaining hardness quotas while round-robining over databases.
    for hardness in HARDNESS_ORDER:
        need = max(0, quotas[hardness])
        dbs = sorted(grouped[hardness])
        while need > 0:
            progressed = False
            for db in dbs:
                bucket = grouped[hardness][db]
                while bucket and int(bucket[-1]["_index"]) in selected_ids:
                    bucket.pop()
                if not bucket:
                    continue
                chosen = bucket.pop()
                selected.append(chosen)
                selected_ids.add(int(chosen["_index"]))
                need -= 1
                progressed = True
                if need == 0:
                    break
            if not progressed:
                break

    if len(selected) < sample_size:
        remaining = [row for row in rows if int(row["_index"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    return sorted(selected[:sample_size], key=lambda row: int(row["_index"]))


def _sqlite_answers(db_path: Path, sql: str) -> list[list[Any]]:
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        return [[_json_value(value) for value in row] for row in con.execute(sql).fetchall()]
    finally:
        con.close()


def _convert_sqlite_to_duckdb(
    sqlite_path: Path,
    duckdb_path: Path,
    force: bool,
    spider_types: dict[tuple[str, str], str] | None = None,
) -> None:
    import duckdb  # Imported lazily so manifest-only validation stays lightweight.
    import pandas as pd

    if duckdb_path.exists():
        if not force:
            return
        duckdb_path.unlink()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    source.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    target = duckdb.connect(str(duckdb_path))
    try:
        tables = [
            row[0]
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        for table in tables:
            qtable = _quote_ident(table)
            columns = source.execute(f"PRAGMA table_info({qtable})").fetchall()
            if not columns:
                continue
            target_types: dict[str, str] = {}
            for column in columns:
                name = str(column[1])
                qname = _quote_ident(name)
                sqlite_types = {
                    str(row[0])
                    for row in source.execute(
                        f"SELECT DISTINCT typeof({qname}) FROM {qtable}"
                    ).fetchall()
                }
                declared = str(column[2] or "")
                official_type = (spider_types or {}).get((table.casefold(), name.casefold()))
                inferred = _duckdb_type(declared, sqlite_types, official_type)
                if inferred in {"BIGINT", "DOUBLE"} and "text" in sqlite_types:
                    nonempty = source.execute(
                        f"SELECT 1 FROM {qtable} WHERE typeof({qname})='text' "
                        f"AND lower(trim(CAST({qname} AS TEXT))) "
                        f"NOT IN ('', 'null', 'none', 'nan') LIMIT 1"
                    ).fetchone()
                    if nonempty and official_type != "number":
                        inferred = "VARCHAR"
                target_types[name] = inferred
            definitions = ", ".join(
                f"{_quote_ident(name)} {target_types[name]}" for name in target_types
            )
            target.execute(f"CREATE TABLE {qtable} ({definitions})")
            # Bulk-copy through pandas instead of DuckDB executemany.  The
            # latter is prohibitively slow for Spider's 510k-row wta_1 table.
            for frame in pd.read_sql_query(f"SELECT * FROM {qtable}", source, chunksize=50_000):
                for name, target_type in target_types.items():
                    if target_type in {"BIGINT", "DOUBLE"}:
                        frame[name] = pd.to_numeric(
                            frame[name].replace(
                                {"": None, "null": None, "NULL": None, "none": None, "None": None, "nan": None, "NaN": None}
                            ),
                            errors="raise",
                        )
                    elif target_type == "VARCHAR":
                        frame[name] = frame[name].map(
                            # pandas represents SQL NULL in string/object
                            # columns as NaN/NA.  Converting that sentinel with
                            # str() silently creates a literal "nan", changing
                            # IS NULL predicates and joins in the target DB.
                            lambda value: (
                                None
                                if value is None or pd.isna(value)
                                else str(_json_value(value))
                            )
                        )
                target.register("_spider_chunk", frame)
                try:
                    target.execute(f"INSERT INTO {qtable} SELECT * FROM _spider_chunk")
                finally:
                    target.unregister("_spider_chunk")
            source_count = source.execute(f"SELECT COUNT(*) FROM {qtable}").fetchone()[0]
            target_count = target.execute(f"SELECT COUNT(*) FROM {qtable}").fetchone()[0]
            if source_count != target_count:
                raise RuntimeError(
                    f"Row-count mismatch for {sqlite_path.name}:{table}: "
                    f"sqlite={source_count}, duckdb={target_count}"
                )
    finally:
        target.close()
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    source_root = _extract_dev_assets(args.archive.resolve(), output_root / "source")
    dev_rows = json.loads((source_root / "dev.json").read_text(encoding="utf-8"))
    if len(dev_rows) != 1034:
        raise RuntimeError(f"Expected 1034 Spider dev rows, found {len(dev_rows)}")

    annotated: list[dict[str, Any]] = []
    for index, row in enumerate(dev_rows):
        item = dict(row)
        item["_index"] = index
        item["complexity"] = _spider_hardness(row["sql"])
        annotated.append(item)

    selected = _select_rows(annotated, args.sample_size, args.seed)

    table_metadata = json.loads((source_root / "tables.json").read_text(encoding="utf-8"))
    spider_type_maps: dict[str, dict[tuple[str, str], str]] = {}
    for database in table_metadata:
        db_id = str(database["db_id"])
        table_names = list(database["table_names_original"])
        type_map: dict[tuple[str, str], str] = {}
        for column_index, (table_index, column_name) in enumerate(database["column_names_original"]):
            if int(table_index) < 0:
                continue
            table_name = str(table_names[int(table_index)])
            type_map[(table_name.casefold(), str(column_name).casefold())] = str(
                database["column_types"][column_index]
            )
        spider_type_maps[db_id] = type_map
    db_ids = sorted({str(row["db_id"]) for row in selected})
    eval_path = output_root / "evaluation.jsonl"
    manifest_path = output_root / "sample_manifest.json"
    database_dir = output_root / "database"
    output_root.mkdir(parents=True, exist_ok=True)

    prepared_rows = []
    for row in selected:
        db_id = str(row["db_id"])
        sqlite_path = source_root / "database" / db_id / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            raise FileNotFoundError(sqlite_path)
        answer = _sqlite_answers(sqlite_path, str(row["query"]))
        prepared_rows.append(
            {
                "question_id": f"spider_dev_{int(row['_index']):04d}",
                "db": db_id,
                "query": str(row["question"]),
                "hint": "",
                "sql": str(row["query"]),
                "answer": answer,
                "complexity": str(row["complexity"]),
                "spider_index": int(row["_index"]),
            }
        )

    with eval_path.open("w", encoding="utf-8") as handle:
        for row in prepared_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    for db_id in db_ids:
        _convert_sqlite_to_duckdb(
            source_root / "database" / db_id / f"{db_id}.sqlite",
            database_dir / f"{db_id}.duckdb",
            args.force,
            spider_type_maps.get(db_id),
        )

    summary = {
        "benchmark": "Spider 1.0 dev",
        "sample_size": len(prepared_rows),
        "seed": args.seed,
        "database_count": len(db_ids),
        "database_ids": db_ids,
        "difficulty_counts": dict(Counter(row["complexity"] for row in prepared_rows)),
        "source_dev_rows": len(dev_rows),
        "archive_sha256": ARCHIVE_SHA256,
        "gold_sql_usage": "reference answers and stratification only; never shown to agents",
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
