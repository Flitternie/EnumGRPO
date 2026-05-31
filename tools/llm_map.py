from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.edit_duckdb import exec_edit_sql
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import (
    _json_safe,
    attach_extra_required_dbs,
    append_edits_log,
    exec_sql_steps_on_conn,
    is_readonly_sql,
    nullsafe_str,
    quote_ident,
    quote_table_name,
    resolve_duckdb_path,
    run_readonly_query_on_conn,
    sql_literal,
    strip_sql_comments,
    validate_type_sql,
    validate_where_clause,
)
from utils.llm import (
    get_llmop_concurrency,
    get_llmop_model,
    heuristic_tokens_from_chars,
    llmop_call,
    llmop_postprocess_output_text,
    payload_for_error,
    plan_fingerprint_payload,
    plan_id,
    strip_wrapping_quotes,
    truncate_cell,
)


def _exec_unlimited_sql(
    *,
    rt: ToolRuntime,
    sql_text: str,
    db_path: Optional[str],
) -> Dict[str, Any]:
    """Execute a read-only SELECT with NO row cap (bypasses the 1000-row limit
    enforced by exec_readonly_sql / run_readonly_query_on_conn).

    Used by llm_map when max_items is None (unlimited mode).
    """
    if not is_readonly_sql(sql_text):
        raise ValueError("input_sql must be a read-only SELECT/WITH statement")

    q = strip_sql_comments(sql_text).strip().rstrip(";").strip()

    if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
        with sess.lock:
            attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
            df = sess.conn.execute(q).fetchdf()
    else:
        import duckdb  # type: ignore

        db_dir = str(rt.db_files_dir or "")
        required = list(rt.required_dbs)
        if db_path:
            primary = db_path
        elif required:
            primary = resolve_duckdb_path(db_dir, required[0]) or ""
        else:
            primary = ""
        if not primary:
            raise RuntimeError("Cannot resolve DuckDB path for unlimited SQL execution")
        con = duckdb.connect(primary, read_only=True)
        try:
            df = con.execute(q).fetchdf()
        finally:
            con.close()

    cols = [str(c) for c in list(df.columns)]
    rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
    return {"columns": cols, "rows": rows, "row_count": int(len(rows))}


# Sentinel distinguishing "key present in cache with NULL output" from "key not in cache at all".
_CACHE_MISS = object()


_NON_SEMANTIC_PLAN_KEYS = {
    # These options affect execution performance/UX, not the semantic work being confirmed.
    "cache_enabled",
    "cache_table",
    "cache_scope",
    "preview_sql",
    "limit_rows",
}


def _plan_fingerprint_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return plan_fingerprint_payload(payload, non_semantic_keys=_NON_SEMANTIC_PLAN_KEYS)


def _coerce_output(text: str, *, output_type: str) -> Any:
    t = strip_wrapping_quotes((text or "").strip())
    if not t:
        return None
    if t.strip().lower() == "null":
        return None
    if output_type == "text":
        return t
    if output_type == "boolean":
        s = t.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
        return None
    if output_type == "number":
        try:
            if re.fullmatch(r"[+-]?\d+", t.strip()):
                return int(t.strip())
            return float(t.strip())
        except Exception:
            return None
    if output_type == "json":
        try:
            return json.loads(t)
        except Exception:
            return None
    return t


def _prompt_for_item(
    *,
    instruction: str,
    value: Any,
    output_type: str,
    options: Optional[str],
    context: Optional[Dict[str, Any]],
) -> str:
    constraints = "Return ONLY the transformed value as plain text."
    if output_type == "boolean":
        constraints = "Return ONLY TRUE or FALSE."
    elif output_type == "number":
        constraints = "Return ONLY a number (no extra text). If unknown, return NULL."
    elif output_type == "json":
        constraints = "Return ONLY a JSON value (object/array/string/number/boolean/null)."

    opts = ""
    if isinstance(options, str) and options.strip():
        opts = f"\nAllowed options (choose one if applicable): {options.strip()}\n"

    ctx = ""
    if isinstance(context, dict) and context:
        try:
            ctx = "\nContext (JSON):\n" + json.dumps(context, ensure_ascii=False) + "\n"
        except Exception:
            ctx = ""

    v = "" if value is None else str(value)
    return (
        "You are transforming database values using an instruction.\n"
        f"{constraints}\n"
        "If the value is invalid/unknown, return NULL.\n"
        f"{opts}"
        "\n"
        f"Instruction:\n{instruction.strip()}\n"
        f"{ctx}"
        "\n"
        f"Input value:\n{v}\n"
    )


def _stable_fingerprint(items: List[Any]) -> str:
    """
    Deterministic list fingerprint for resume/safety.
    """
    try:
        blob = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    except Exception:
        blob = str(items).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()[:16]


_SIMPLE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: Any, *, field: str) -> str:
    s = str(name or "").strip()
    if not s:
        raise ValueError(f"{field} is required")
    if not _SIMPLE_IDENT_RE.match(s):
        raise ValueError(f"Invalid {field}: {s!r}")
    return s


def _cache_ddl(cache_table: str, *, temp: bool) -> str:
    tbl = quote_ident(cache_table)
    prefix = "CREATE TEMP TABLE IF NOT EXISTS" if temp else "CREATE TABLE IF NOT EXISTS"
    return (
        f"{prefix} {tbl} ("
        "map_id VARCHAR NOT NULL, "
        "input_text VARCHAR NOT NULL, "
        "ctx_hash VARCHAR NOT NULL, "
        "output_text VARCHAR, "
        "created_ts TIMESTAMP DEFAULT now(), "
        "PRIMARY KEY(map_id, input_text, ctx_hash)"
        ");"
    )


def _cache_select_sql(cache_table: str, *, map_id: str, keys: List[Tuple[str, str]]) -> str:
    """
    keys: [(input_text, ctx_hash), ...]
    Returns columns: (input_text, ctx_hash, output_text, cache_hit).
    cache_hit is TRUE when a matching cache row exists (output_text may still be NULL for cached-NULL outputs).
    """
    tbl = quote_ident(cache_table)
    values_rows = ",\n".join(f"({sql_literal(k)}, {sql_literal(h)})" for (k, h) in keys)
    values_cte = f"(VALUES\n{values_rows}\n) AS q(input_text, ctx_hash)"
    return (
        f"SELECT q.input_text, q.ctx_hash, c.output_text, "
        f"(c.map_id IS NOT NULL) AS cache_hit "
        f"FROM {values_cte} "
        f"LEFT JOIN {tbl} AS c "
        f"ON c.map_id = {sql_literal(map_id)} AND c.input_text = q.input_text AND c.ctx_hash = q.ctx_hash"
    )


def _cache_upsert_sql(cache_table: str, *, map_id: str, rows: List[Tuple[str, str, Optional[str]]]) -> List[str]:
    """
    rows: [(input_text, ctx_hash, output_text), ...]
    """
    tbl = quote_ident(cache_table)
    stmts: List[str] = []
    for chunk in _chunk_pairs([(a, b, c) for (a, b, c) in rows], 500):
        values_rows = ",\n".join(
            f"({sql_literal(a)}, {sql_literal(b)}, {sql_literal(c)})" for (a, b, c) in chunk
        )
        values_cte = f"(VALUES\n{values_rows}\n) AS v(input_text, ctx_hash, output_text)"
        stmts.append(
            f"INSERT INTO {tbl} (map_id, input_text, ctx_hash, output_text) "
            f"SELECT {sql_literal(map_id)} AS map_id, v.input_text, v.ctx_hash, v.output_text "
            f"FROM {values_cte} "
            f"ON CONFLICT (map_id, input_text, ctx_hash) DO UPDATE SET "
            f"output_text = excluded.output_text, created_ts = now();"
        )
    return stmts


def _llm_map_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "llm_map",
            "description": (
                "Fetch values with a SQL subquery, have an LLM annotate each row, then write results back into a table column — in one call. "
                "Use instead of iterating SQL when rows need per-row classification, normalization, or world-knowledge annotation. "
                "Pass the filtering SQL via input_sql; call materialize_temp when the intermediate result is reused across multiple steps. "
                "action='run' (default) executes immediately; action='plan' previews token cost only. "
                "Can write directly into a TEMP table created earlier in the same session."
            ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": ["string", "null"],
                    "enum": ["plan", "run", None],
                    "description": "Default 'run'. Use 'plan' to preview token cost before executing.",
                },
                "instruction": {"type": "string", "description": "The mapping instruction (e.g. 'What country is this city in?')."},
                "distinct_from": {
                    "type": ["object", "null"],
                    "description": "Fetch DISTINCT values from a table column (cheapest input).",
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                        "where": {"type": ["string", "null"]},
                    },
                    "required": ["table", "column"],
                    "additionalProperties": False,
                },
                "input_sql": {
                    "type": ["string", "null"],
                    "description": (
                        "Read-only SELECT/WITH as input — pass your filtering SQL here. "
                        "mode=distinct: 1 column (value). "
                        "mode=per_row: (row_id, value[, ctx...])."
                    ),
                },
                "mode": {
                    "type": ["string", "null"],
                    "enum": ["distinct", "per_row", None],
                    "description": "distinct (default) or per_row mapping.",
                },
                "key_strategy": {
                    "type": ["string", "null"],
                    "enum": ["by_value", "by_row_id", None],
                    "description": "Join results by value (default) or by row_id (use with per_row).",
                },
                "target": {
                    "type": ["object", "null"],
                    "description": "Where to write mapped outputs. In a read-only session, if the target table is not a TEMP table, results are automatically written to a new TEMP TABLE named llm_map_<target_column> that you can JOIN against.",
                    "properties": {
                        "table": {"type": "string"},
                        "key_column": {"type": ["string", "null"]},
                        "target_column": {"type": "string"},
                        "create_column": {"type": ["boolean", "null"]},
                        "new_column_type": {"type": ["string", "null"]},
                    },
                    "required": ["table", "target_column"],
                    "additionalProperties": False,
                },
                "output_type": {
                    "type": ["string", "null"],
                    "enum": ["text", "number", "boolean", "json", None],
                    "description": "Expected output type (default: text).",
                },
                "options": {"type": ["string", "null"], "description": "Allowed output values / rubric."},
                "max_items": {"type": ["integer", "null"], "description": "Max distinct values to process (default 200)."},
                "max_cell_chars": {"type": ["integer", "null"], "description": "Max chars per input cell sent to LLM (default 120)."},
                "max_ctx_rows": {"type": ["integer", "null"], "description": "Per-row mode: max extra context columns from input_sql (default 8)."},
                "confirm": {"type": ["boolean", "null"], "description": "Required only for two-step plan→run flow."},
                "plan_id": {"type": ["string", "null"], "description": "Required only for two-step plan→run flow."},
                "db_path": {"type": ["string", "null"], "description": "Optional: DB path for fetching input."},
                # Legacy / advanced params kept in handler but hidden from schema:
                # table, column_name, where, target_column, create_column, new_column_type,
                # write_mode, resume_offset, batch_size, unmapped_behavior, options_sql,
                # cache, target_db_path, dest_db_name, preview_sql, limit_rows
            },
            "required": ["instruction"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _fetch_distinct_values(
    *,
    rt: ToolRuntime,
    table_sql: str,
    col_sql: str,
    where: Optional[str],
    max_items: Optional[int],
    db_path: Optional[str],
) -> List[Any]:
    sql = f"SELECT DISTINCT {col_sql} AS v FROM {table_sql}"
    if where:
        sql += f" WHERE {where}"
    if max_items is not None:
        sql += f" LIMIT {int(max_items)}"
    if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
        with sess.lock:
            attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
            res = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql, limit_rows=max_items)
        rows = res.get("rows") or []
        out: List[Any] = []
        for r in rows:
            if isinstance(r, list) and r:
                out.append(r[0])
        return out
    res = exec_readonly_sql(
        sql_text=sql,
        required_dbs=list(rt.required_dbs),
        db_files_dir=str(rt.db_files_dir or ""),
        db_edit_dir=str(rt.db_edit_dir or ""),
        db_path=db_path,
        limit_rows=max_items,
    )
    # rows: [[v], ...]
    rows = res.get("rows") or []
    out: List[Any] = []
    for r in rows:
        if isinstance(r, list) and r:
            out.append(r[0])
    return out


def _llm_transform_value(*, model: str, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
    out_text, usage = llmop_call(model=model, prompt=prompt)
    out = llmop_postprocess_output_text(out_text or "")
    if not out:
        return (None, usage)
    if out.strip().lower() == "null":
        return (None, usage)
    return (out, usage)


def _chunk_pairs(pairs: List[Tuple[Any, Any]], n: int) -> List[List[Tuple[Any, Any]]]:
    return [pairs[i : i + n] for i in range(0, len(pairs), n)]


def _llm_map_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        action = str(args.get("action") or "run").strip().lower()
        if action not in ("plan", "run"):
            raise ValueError("action must be 'plan' or 'run'")

        # Input selection
        distinct_from = args.get("distinct_from")
        input_sql = nullsafe_str(args.get("input_sql"))

        mode_raw = args.get("mode")
        mode = "distinct" if mode_raw is None else str(mode_raw).strip().lower()
        if mode not in ("distinct", "per_row"):
            raise ValueError("mode must be 'distinct', 'per_row', or null")

        key_strategy_raw = args.get("key_strategy")
        key_strategy = "by_value" if key_strategy_raw is None else str(key_strategy_raw).strip().lower()
        if key_strategy not in ("by_value", "by_row_id"):
            raise ValueError("key_strategy must be 'by_value', 'by_row_id', or null")

        # Determine output target
        target = args.get("target")
        if not isinstance(target, dict):
            raise ValueError("target is required (provide target.table and target.target_column).")

        target_table_raw = target.get("table")
        target_table_sql = quote_table_name(str(target_table_raw or "").strip())
        target_key_col_raw = target.get("key_column")
        target_key_col = (
            quote_ident(str(target_key_col_raw).strip())
            if isinstance(target_key_col_raw, str) and target_key_col_raw.strip()
            else None
        )
        target_col_raw = str(target.get("target_column") or "").strip()
        if not target_col_raw:
            raise ValueError("target.target_column is required")
        target_col_sql = quote_ident(target_col_raw)
        create_col = target.get("create_column")
        create = bool(create_col) if create_col is not None else False
        type_sql = validate_type_sql(target.get("new_column_type")) if create else ""

        write_mode_raw = args.get("write_mode")
        write_mode = "fill_missing" if write_mode_raw is None else str(write_mode_raw).strip().lower()
        if write_mode not in ("fill_missing", "overwrite"):
            raise ValueError("write_mode must be 'fill_missing', 'overwrite', or null")

        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        if len(instruction) > 2000:
            raise ValueError("instruction is too long (max 2000 chars)")

        max_items = args.get("max_items")
        # None means unlimited — let the caller's input_sql control row count.
        mi: Optional[int] = int(max_items) if max_items is not None else None
        if mi is not None:
            mi = max(1, mi)

        unmapped = args.get("unmapped_behavior")
        unmapped_behavior = "keep" if unmapped is None else str(unmapped).strip().lower()
        if unmapped_behavior not in ("keep", "null"):
            raise ValueError("unmapped_behavior must be 'keep', 'null', or null")

        db_path = args.get("db_path")
        db_path_norm = str(db_path).strip() if isinstance(db_path, str) and db_path.strip() else None

        output_type_raw = args.get("output_type")
        output_type = "text" if output_type_raw is None else str(output_type_raw).strip().lower()
        if output_type not in ("text", "number", "boolean", "json"):
            raise ValueError("output_type must be one of: text, number, boolean, json (or null)")

        max_cell_chars_raw = args.get("max_cell_chars")
        max_cell_chars = int(max_cell_chars_raw) if max_cell_chars_raw is not None else 120
        max_cell_chars = max(10, min(max_cell_chars, 500))

        max_ctx_cols_raw = args.get("max_ctx_rows")
        max_ctx_cols = int(max_ctx_cols_raw) if max_ctx_cols_raw is not None else 8
        max_ctx_cols = max(0, min(max_ctx_cols, 20))

        # Resolve options/options_sql
        options = args.get("options")
        options_s = str(options).strip() if isinstance(options, str) and options.strip() else ""
        options_sql = nullsafe_str(args.get("options_sql"))

        def _fetch_options_from_sql(sql_text: str) -> str:
            # read-only fetch; expects 1 column
            if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
                sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
                with sess.lock:
                    attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                    out = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql_text, limit_rows=200)
            else:
                out = exec_readonly_sql(
                    sql_text=sql_text,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    db_path=db_path_norm,
                    limit_rows=200,
                )
            cols = out.get("columns") or []
            rows = out.get("rows") or []
            if not cols:
                return ""
            vals: List[str] = []
            for r in rows:
                if isinstance(r, list) and r:
                    if r[0] is None:
                        continue
                    vals.append(str(r[0]))
            # Keep it compact: semicolon separated
            return ";".join(vals[:200])

        if options_sql:
            options_s = _fetch_options_from_sql(options_sql)

        # Fetch inputs
        input_rows: List[List[Any]] = []
        input_cols: List[str] = []
        if input_sql:
            if not isinstance(input_sql, str) or not input_sql.strip():
                raise ValueError("input_sql must be a string or null")
            if not re.match(r"(?is)^\s*(select|with)\b", input_sql.strip()):
                raise ValueError("input_sql must be read-only SELECT/WITH")
            if mi is None:
                out = _exec_unlimited_sql(rt=rt, sql_text=input_sql, db_path=db_path_norm)
            elif getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
                sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
                with sess.lock:
                    attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                    out = run_readonly_query_on_conn(conn=sess.conn, sql_text=input_sql, limit_rows=mi)
            else:
                out = exec_readonly_sql(
                    sql_text=input_sql,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    db_path=db_path_norm,
                    limit_rows=mi,
                )
            input_cols = [str(c) for c in (out.get("columns") or [])]
            input_rows = [r for r in (out.get("rows") or []) if isinstance(r, list)]
        else:
            if isinstance(distinct_from, dict):
                src_table = quote_table_name(str(distinct_from.get("table") or "").strip())
                src_col = quote_ident(str(distinct_from.get("column") or "").strip())
                src_where = validate_where_clause(distinct_from.get("where"))
            else:
                raise ValueError("Either input_sql or distinct_from must be provided.")
            values = _fetch_distinct_values(
                rt=rt,
                table_sql=src_table,
                col_sql=src_col,
                where=src_where,
                max_items=mi,
                db_path=db_path_norm,
            )
            input_cols = ["value"]
            input_rows = [[v] for v in values]

        # Normalize into mapping items
        items_by_value: List[Any] = []
        items_by_row: List[Tuple[Any, Any, Dict[str, Any]]] = []
        if mode == "distinct":
            # expect one column: value
            for r in input_rows:
                if not r:
                    continue
                items_by_value.append(r[0])
        else:
            # per_row expects at least row_id, value
            for r in input_rows:
                if len(r) < 2:
                    continue
                row_id = r[0]
                value = r[1]
                ctx: Dict[str, Any] = {}
                # include up to max_ctx_cols extra columns
                extra = r[2 : 2 + max_ctx_cols]
                for idx, v in enumerate(extra, start=2):
                    colname = input_cols[idx] if idx < len(input_cols) else f"ctx{idx-1}"
                    ctx[colname] = truncate_cell(v, max_chars=max_cell_chars)
                items_by_row.append((row_id, value, ctx))

        # Only send non-null values to the LLM.
        non_null_values = [v for v in items_by_value if v is not None]
        non_null_rows = [(rid, v, ctx) for (rid, v, ctx) in items_by_row if v is not None]

        model_name = get_llmop_model()

        # Cache config
        cache_cfg = args.get("cache")
        cache_enabled = True
        cache_table = "__llm_map_cache"
        cache_scope = "session"
        if isinstance(cache_cfg, dict):
            cache_enabled = bool(cache_cfg.get("enabled")) if cache_cfg.get("enabled") is not None else True
            if isinstance(cache_cfg.get("cache_table"), str) and str(cache_cfg.get("cache_table")).strip():
                cache_table = _validate_identifier(cache_cfg.get("cache_table"), field="cache.cache_table")
            scope_raw = cache_cfg.get("scope")
            cache_scope = "session" if scope_raw is None else str(scope_raw).strip().lower()
            if cache_scope not in ("session", "db"):
                raise ValueError("cache.scope must be 'session', 'db', or null")

        # Map id for cache + resume fingerprint
        map_id_payload = {
            "model": model_name,
            "instruction": instruction,
            "output_type": output_type,
            "options": options_s or "",
            "mode": mode,
            "key_strategy": key_strategy,
        }
        map_id = plan_id(map_id_payload)

        # Determine deterministic missing-items list & token estimate
        missing_items: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]] = []
        # Each entry: (input_text, ctx_hash, key_for_write, ctx_for_prompt)
        if mode == "distinct":
            ordered = sorted({str(v) for v in non_null_values})
            for s in ordered:
                missing_items.append((s, "", s, None))
        else:
            # per_row
            ordered_rows = sorted(non_null_rows, key=lambda t: (str(t[0]), str(t[1])))
            for rid, v, ctx in ordered_rows:
                input_text = str(v)
                ctx_hash = hashlib.sha256(json.dumps(ctx, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16] if ctx else ""
                key_for_write = rid if key_strategy == "by_row_id" else input_text
                missing_items.append((input_text, ctx_hash, key_for_write, {"row_id": rid, **ctx}))

        fingerprint = _stable_fingerprint([(a, b) for (a, b, _k, _c) in missing_items])

        # Estimate prompt size for first N items (upper bound)
        sample_for_est = missing_items[: min(50, len(missing_items))]
        est_chars = 0
        for input_text, _ctx_hash, _key, ctx in sample_for_est:
            p = _prompt_for_item(
                instruction=instruction,
                value=truncate_cell(input_text, max_chars=max_cell_chars),
                output_type=output_type,
                options=options_s,
                context=ctx,
            )
            est_chars += len(p)
        est_tokens = heuristic_tokens_from_chars(est_chars) * (len(missing_items) / max(1, len(sample_for_est)))

        plan_payload = {
            "map_id": map_id,
            "fingerprint": fingerprint,
            "mode": mode,
            "key_strategy": key_strategy,
            "write_mode": write_mode,
            "target_table": str(target_table_sql),
            "target_column": str(target_col_raw),
            "create_column": bool(create),
            "new_column_type": type_sql or "",
            "max_items": int(mi) if mi is not None else None,            "max_cell_chars": int(max_cell_chars),
            "max_ctx_cols": int(max_ctx_cols),
            "cache_enabled": bool(cache_enabled),
            "cache_table": cache_table,
            "cache_scope": cache_scope,
            "n_items": int(len(missing_items)),
        }
        fp_payload = _plan_fingerprint_payload(plan_payload)
        pid = plan_id(fp_payload)

        if action == "plan":
            n_cached: Optional[int] = None
            n_to_call: Optional[int] = None
            if cache_enabled and cache_scope in ("session", "db") and getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
                sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
                with sess.lock:
                    attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                    if cache_scope == "db" and getattr(sess, "read_only", False):
                        raise ValueError("cache.scope='db' requires a writable session/DB under db_edit_dir.")
                    sess.conn.execute(_cache_ddl(cache_table, temp=(cache_scope == "session")))
                    keys = [(a, b) for (a, b, _k, _c) in missing_items]
                    if keys:
                        sql = _cache_select_sql(cache_table, map_id=map_id, keys=keys)
                        out = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql, limit_rows=len(keys))
                        rows = out.get("rows") or []
                        hits = 0
                        for r in rows:
                            if isinstance(r, list) and len(r) >= 4 and r[3]:
                                hits += 1
                        n_cached = int(hits)
                        n_to_call = int(max(0, len(keys) - hits))
            return {
                "action": "plan",
                "plan_id": pid,
                "model": model_name,
                "mode": mode,
                "key_strategy": key_strategy,
                "write_mode": write_mode,
                "fingerprint": fingerprint,
                "n_rows": int(len(non_null_rows)) if mode == "per_row" else None,
                "n_distinct": int(len(set(non_null_values))) if mode == "distinct" else None,
                "n_items": int(len(missing_items)),
                "n_cached": n_cached,
                "n_to_call": n_to_call,
                "max_items": int(mi) if mi is not None else None,                "plan_fingerprint_payload": fp_payload,
                "estimated_tokens_upper_bound": int(math.ceil(est_tokens)),
                "sample_items": [{"value": a, "ctx_hash": b} for (a, b, _k, _c) in missing_items[:20]],
                "note": "To execute, call again with action='run', confirm=true, and the same plan_id.",
            }

        # action == "run"
        plan_id_in = args.get("plan_id")
        if isinstance(plan_id_in, str) and plan_id_in.strip():
            # Two-step flow: validate confirm + plan_id fingerprint.
            confirm = args.get("confirm")
            if confirm is not True:
                raise ValueError("When plan_id is provided, confirm=true is required.")
            if plan_id_in.strip() != pid:
                raise ValueError(
                    "plan_id mismatch.\n"
                    f"- provided: {plan_id_in.strip()}\n"
                    f"- expected: {pid}\n"
                    f"- current_plan_fingerprint_payload: {payload_for_error(fp_payload)}\n"
                    "Re-run action='plan' and use the returned plan_id."
                )

        max_calls_raw = args.get("max_calls")
        max_calls = int(max_calls_raw) if max_calls_raw is not None else 2000
        max_calls = max(1, min(max_calls, 2000))

        resume_offset_raw = args.get("resume_offset")
        resume_offset = int(resume_offset_raw) if resume_offset_raw is not None else 0
        resume_offset = max(0, min(resume_offset, len(missing_items)))

        concurrency = get_llmop_concurrency()

        to_process = missing_items[resume_offset : resume_offset + max_calls]

        mapping_pairs: List[Tuple[Any, Any]] = []
        mapping_pairs_by_row: List[Tuple[Any, Any]] = []
        failures: List[Dict[str, Any]] = []
        cache_hits: Dict[Tuple[str, str], Optional[str]] = {}
        cache_misses: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]] = []

        # If caching enabled, attempt to read cache (session/db scope in session mode).
        if cache_enabled and cache_scope in ("session", "db") and getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                if cache_scope == "db" and getattr(sess, "read_only", False):
                    raise ValueError("cache.scope='db' requires a writable session/DB under db_edit_dir.")
                sess.conn.execute(_cache_ddl(cache_table, temp=(cache_scope == "session")))
                keys = [(a, b) for (a, b, _k, _c) in to_process]
                if keys:
                    sql = _cache_select_sql(cache_table, map_id=map_id, keys=keys)
                    out = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql, limit_rows=len(keys))
                    rows = out.get("rows") or []
                    for r in rows:
                        if isinstance(r, list) and len(r) >= 4 and r[3]:
                            # r[3] is cache_hit; r[2] is output_text (may be None for cached-NULL outputs)
                            cache_hits[(str(r[0]), str(r[1]))] = r[2]

        for input_text, ctx_hash, key_for_write, ctx in to_process:
            cached = cache_hits.get((input_text, ctx_hash), _CACHE_MISS)
            if cached is not _CACHE_MISS:
                # Genuine cache hit — output_text may be None (cached-NULL); skip LLM call.
                if cached is not None:
                    out_text = str(cached)
                    if key_strategy == "by_row_id" and mode == "per_row":
                        mapping_pairs_by_row.append((key_for_write, out_text))
                    else:
                        mapping_pairs.append((input_text, out_text))
                continue
            cache_misses.append((input_text, ctx_hash, key_for_write, ctx))

        cache_upserts: List[Tuple[str, str, Optional[str]]] = []

        progress_cb = getattr(rt, "progress_callback", None)
        total_llm_calls = len(cache_misses)
        if progress_cb and total_llm_calls > 0:
            try:
                progress_cb(0, float(total_llm_calls), f"Starting LLM mapping: {total_llm_calls} calls")
            except Exception:
                pass

        def _do_one_call(
            item: Tuple[str, str, Any, Optional[Dict[str, Any]]],
        ) -> Tuple[str, str, Any, Optional[str], Optional[str], Optional[Dict[str, int]]]:
            """Run a single LLM call. Returns (input_text, ctx_hash, key_for_write, result_text_or_None, error_or_None, token_usage_or_None)."""
            input_text, ctx_hash, key_for_write, ctx = item
            prompt = _prompt_for_item(
                instruction=instruction,
                value=truncate_cell(input_text, max_chars=max_cell_chars),
                output_type=output_type,
                options=options_s,
                context=ctx,
            )
            try:
                raw, usage = _llm_transform_value(model=model_name, prompt=prompt)
            except Exception as e:
                return (input_text, ctx_hash, key_for_write, None, str(e), None)
            if raw is None:
                return (input_text, ctx_hash, key_for_write, None, None, usage)
            typed = _coerce_output(raw, output_type=output_type)
            if typed is None and output_type != "text":
                return (input_text, ctx_hash, key_for_write, None, f"output_type coercion failed|{raw}", usage)
            out_text = raw if output_type == "text" else str(typed)
            return (input_text, ctx_hash, key_for_write, out_text, None, usage)

        completed_count = 0
        completed_lock = threading.Lock()

        token_usage_total: Dict[str, int] = {}
        token_usage_reports = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_do_one_call, item): item for item in cache_misses}
            for future in as_completed(futures):
                input_text_r, ctx_hash_r, key_for_write_r, out_text, error, usage = future.result()
                if error is not None:
                    if error.startswith("output_type coercion failed|"):
                        raw_val = error.split("|", 1)[1]
                        failures.append({"from": input_text_r, "ctx_hash": ctx_hash_r, "error": "output_type coercion failed", "raw": raw_val})
                    else:
                        failures.append({"from": input_text_r, "ctx_hash": ctx_hash_r, "error": error})
                else:
                    if out_text is not None:
                        if key_strategy == "by_row_id" and mode == "per_row":
                            mapping_pairs_by_row.append((key_for_write_r, out_text))
                        else:
                            mapping_pairs.append((input_text_r, out_text))
                    # Cache the result even when out_text is None (LLM returned NULL) so we
                    # don't re-call the LLM for the same input on a resumed or repeated run.
                    if cache_enabled:
                        cache_upserts.append((input_text_r, ctx_hash_r, out_text))

                if isinstance(usage, dict) and usage:
                    token_usage_reports += 1
                    for k, v in usage.items():
                        try:
                            token_usage_total[k] = int(token_usage_total.get(k, 0) + int(v))
                        except Exception:
                            pass

                with completed_lock:
                    completed_count += 1
                if progress_cb:
                    try:
                        progress_cb(float(completed_count), float(total_llm_calls), f"Mapped {completed_count}/{total_llm_calls} values")
                    except Exception:
                        pass

        if progress_cb and total_llm_calls > 0:
            try:
                progress_cb(float(total_llm_calls), float(total_llm_calls), "LLM mapping complete")
            except Exception:
                pass

        # Build SQL steps for write-back
        sql_steps: List[str] = []
        if create:
            sql_steps.append(f"ALTER TABLE {target_table_sql} ADD COLUMN {target_col_sql} {type_sql};")

        # Optional additional scope filter 
        scope_where: Optional[str] = None
        if not input_sql:
            # distinct_from or legacy selection filter should also constrain updates.
            if isinstance(distinct_from, dict):
                scope_where = validate_where_clause(distinct_from.get("where"))

        where_write_parts: List[str] = []
        if write_mode == "fill_missing":
            where_write_parts.append(f"{target_col_sql} IS NULL")
        if scope_where:
            where_write_parts.append(f"({scope_where})")
        where_write = " AND ".join(where_write_parts) if where_write_parts else None

        if mode == "per_row" and key_strategy == "by_row_id":
            if target_key_col is None:
                raise ValueError("target.key_column is required when key_strategy=by_row_id")
            # Apply per-row updates by key column
            for chunk in _chunk_pairs(mapping_pairs_by_row, 1000):
                values_rows = ",\n".join(f"({sql_literal(k)}, {sql_literal(v)})" for (k, v) in chunk)
                values_cte = f"(VALUES\n{values_rows}\n) AS m(src, dst)"
                stmt = (
                    f"UPDATE {target_table_sql} AS t SET {target_col_sql} = m.dst "
                    f"FROM {values_cte} WHERE t.{target_key_col} = m.src"
                )
                if where_write:
                    stmt += f" AND ({where_write})"
                stmt += ";"
                sql_steps.append(stmt)
        else:
            # Value-based mapping (distinct or per_row by_value)
            # Determine source column to match against (defaults to legacy/distinct_from column; fallback to target column).
            source_col_sql = None
            if isinstance(distinct_from, dict):
                src_col_raw = str(distinct_from.get("column") or "").strip()
                source_col_sql = quote_ident(src_col_raw) if src_col_raw else None
            if source_col_sql is None:
                source_col_sql = target_col_sql

            writing_new_col = target_col_sql != source_col_sql

            # Initialize target values for unmapped rows, matching prior behavior.
            # where_write_mapped: the WHERE condition to append to the mapped-values UPDATE.
            if writing_new_col and unmapped_behavior == "keep":
                # The init fills ALL in-scope rows with the source column as a placeholder
                # for unmapped rows.  It must NOT use the fill_missing IS NULL guard:
                # the column is freshly added (all NULL), so the guard would be vacuously
                # satisfied now but would block the mapped update after the init runs.
                init_where = f"({scope_where})" if scope_where else None
                init = f"UPDATE {target_table_sql} SET {target_col_sql} = {source_col_sql}"
                if init_where:
                    init += f" WHERE {init_where}"
                init += ";"
                sql_steps.append(init)
                # After init, the column is non-NULL for every in-scope row.
                # The mapped update must overwrite the placeholder without an IS NULL guard.
                where_write_mapped = init_where
            elif unmapped_behavior == "null":
                wipe = f"UPDATE {target_table_sql} SET {target_col_sql} = NULL"
                if where_write:
                    wipe += f" WHERE {where_write}"
                wipe += ";"
                sql_steps.append(wipe)
                where_write_mapped = where_write
            else:
                where_write_mapped = where_write

            for chunk in _chunk_pairs(mapping_pairs, 1000):
                values_rows = ",\n".join(f"({sql_literal(k)}, {sql_literal(v)})" for (k, v) in chunk)
                values_cte = f"(VALUES\n{values_rows}\n) AS m(src, dst)"
                stmt = (
                    f"UPDATE {target_table_sql} AS t SET {target_col_sql} = m.dst "
                    f"FROM {values_cte} WHERE t.{source_col_sql} = m.src"
                )
                if where_write_mapped:
                    stmt += f" AND ({where_write_mapped})"
                stmt += ";"
                sql_steps.append(stmt)

        # Cache write-back (session/db scope in session mode)
        if cache_enabled and cache_scope in ("session", "db") and cache_upserts and getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                if cache_scope == "db" and getattr(sess, "read_only", False):
                    raise ValueError("cache.scope='db' requires a writable session/DB under db_edit_dir.")
                sess.conn.execute(_cache_ddl(cache_table, temp=(cache_scope == "session")))
                for stmt in _cache_upsert_sql(cache_table, map_id=map_id, rows=cache_upserts):
                    sess.conn.execute(stmt)

        limit_rows = args.get("limit_rows")
        limit = int(limit_rows) if limit_rows is not None else 200

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    read_only=True,
                )

                # Read-only session: check whether the target table is a TEMP table.
                # If it is not, redirect output to a TEMP table instead of trying to
                # UPDATE the base table (which would fail with a permission error).
                is_read_only_sess = getattr(sess, "read_only", False)
                target_is_temp = False
                if is_read_only_sess and sql_steps:
                    try:
                        res = sess.conn.execute(
                            "SELECT table_schema FROM information_schema.tables "
                            f"WHERE table_name = {sql_literal(str(target_table_raw or '').strip().lower())} "
                            "LIMIT 1"
                        ).fetchone()
                        target_is_temp = res is not None and str(res[0]).lower() == "temp"
                    except Exception:
                        target_is_temp = False

                if is_read_only_sess and not target_is_temp:
                    # Build a TEMP output table: CREATE TEMP TABLE llm_map_<col> AS
                    # SELECT key_col, dst AS target_col FROM (VALUES …).
                    # This avoids any UPDATE on the read-only base table.
                    safe_col = re.sub(r"\W+", "_", target_col_raw)
                    temp_out = f"llm_map_{safe_col}"
                    temp_out_q = quote_ident(temp_out)

                    if mode == "per_row" and key_strategy == "by_row_id" and mapping_pairs_by_row:
                        pairs = mapping_pairs_by_row
                        key_col_name = str(target_key_col_raw or "key").strip() if target_key_col_raw else "key"
                    else:
                        pairs = mapping_pairs
                        if isinstance(distinct_from, dict):
                            _src = str(distinct_from.get("column") or "").strip()
                            key_col_name = _src if _src else target_col_raw
                        elif input_sql and input_cols:
                            key_col_name = str(input_cols[0])
                        else:
                            key_col_name = target_col_raw

                    if pairs:
                        key_col_q = quote_ident(key_col_name)
                        values_rows = ",\n".join(
                            f"({sql_literal(k)}, {sql_literal(v)})" for (k, v) in pairs
                        )
                        create_stmt = (
                            f"CREATE OR REPLACE TEMP TABLE {temp_out_q} AS "
                            f"SELECT m.key AS {key_col_q}, m.val AS {target_col_sql} "
                            f"FROM (VALUES\n{values_rows}\n) AS m(key, val);"
                        )
                        sql_log_ro, statements_ro = exec_sql_steps_on_conn(
                            conn=sess.conn, sql_steps=[create_stmt]
                        )
                    else:
                        sql_log_ro, statements_ro = [], []
                        temp_out = ""

                    preview = None
                    preview_sql = args.get("preview_sql")
                    if not preview_sql and temp_out:
                        preview_sql = f"SELECT * FROM {temp_out_q} LIMIT 20"
                    if isinstance(preview_sql, str) and preview_sql.strip() and temp_out:
                        try:
                            preview = run_readonly_query_on_conn(
                                conn=sess.conn, sql_text=preview_sql, limit_rows=limit
                            )
                        except Exception:
                            preview = None

                    out: Dict[str, Any] = {
                        "ok": True,
                        "edited_db_path": str(getattr(sess, "db_path", "") or ""),
                        "cloned_from": None,
                        "attached_db_files": [],
                        "sql_log": sql_log_ro,
                        "statements_executed": statements_ro,
                        "preview": preview,
                        "note": (
                            f"Read-only session: results written to TEMP TABLE {temp_out!r} "
                            f"(columns: {key_col_name}, {target_col_raw}). "
                            f"Next step: SELECT original.*, m.{target_col_raw} FROM original_table JOIN {temp_out!r} m USING ({key_col_name}) ..."
                            if temp_out else
                            "Read-only session: no rows to write (empty mapping)."
                        ),
                    }
                    out["llm_map"] = {
                        "plan_id": pid,
                        "fingerprint": fingerprint,
                        "model": model_name,
                        "mode": mode,
                        "key_strategy": key_strategy,
                        "write_mode": write_mode,
                        "output_type": output_type,
                        "options_sql_used": bool(options_sql),
                        "cache": {"enabled": bool(cache_enabled), "scope": cache_scope, "cache_table": cache_table},
                        "resume_offset": int(resume_offset),
                        "next_offset": int(resume_offset + len(to_process)),
                        "done": bool(resume_offset + len(to_process) >= len(missing_items)),
                        "n_items": int(len(missing_items)),
                        "n_processed": int(len(to_process)),
                        "n_cache_hits": int(len([1 for (a, b, _k, _c) in to_process if (a, b) in cache_hits])),
                        "n_called": int(len(cache_misses)),
                        "n_mapped": int(len(mapping_pairs) + len(mapping_pairs_by_row)),
                        "token_usage": token_usage_total or None,
                        "token_usage_reports": int(token_usage_reports),
                        "failures": failures[:50],
                        "temp_output_table": temp_out or None,
                    }
                    out["mapping_preview"] = [{"from": k, "to": v} for (k, v) in mapping_pairs[:100]]
                    return out

                try:
                    sql_log, statements = exec_sql_steps_on_conn(conn=sess.conn, sql_steps=sql_steps)
                except Exception as e:
                    # In read-only sessions, DuckDB may still allow TEMP tables/objects to be modified.
                    # If this fails, provide a more actionable message.
                    if getattr(sess, "read_only", False):
                        raise ValueError(
                            "llm_map could not write results back to the table. "
                            "The session is read-only, so llm_map can only annotate TEMP tables "
                            "created in the same session (not persistent base tables). "
                            "Fix: use materialize_temp() to create a TEMP copy of the table first, "
                            "then run llm_map on that TEMP table. Do NOT call close_session."
                        ) from e
                    raise
                append_edits_log(
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    edited_db_path=str(getattr(sess, "db_path", "") or ""),
                    sql_log=sql_log,
                )
                preview_sql = args.get("preview_sql")
                preview = None
                if isinstance(preview_sql, str) and preview_sql.strip():
                    preview = run_readonly_query_on_conn(conn=sess.conn, sql_text=preview_sql, limit_rows=limit)
            out: Dict[str, Any] = {
                "ok": True,
                "edited_db_path": str(getattr(sess, "db_path", "") or ""),
                "cloned_from": None,
                "attached_db_files": [],
                "sql_log": sql_log,
                "statements_executed": statements,
                "preview": preview,
                "note": "Session mode: llm_map applied updates directly on the existing session connection.",
            }
            out["llm_map"] = {
                "plan_id": pid,
                "fingerprint": fingerprint,
                "model": model_name,
                "mode": mode,
                "key_strategy": key_strategy,
                "write_mode": write_mode,
                "output_type": output_type,
                "options_sql_used": bool(options_sql),
                "cache": {"enabled": bool(cache_enabled), "scope": cache_scope, "cache_table": cache_table},
                "resume_offset": int(resume_offset),
                "next_offset": int(resume_offset + len(to_process)),
                "done": bool(resume_offset + len(to_process) >= len(missing_items)),
                "n_items": int(len(missing_items)),
                "n_processed": int(len(to_process)),
                "n_cache_hits": int(len([1 for (a, b, _k, _c) in to_process if (a, b) in cache_hits])),
                "n_called": int(len(cache_misses)),
                "n_mapped": int(len(mapping_pairs) + len(mapping_pairs_by_row)),
                "token_usage": token_usage_total or None,
                "token_usage_reports": int(token_usage_reports),
                "failures": failures[:50],
            }
            out["mapping_preview"] = [{"from": k, "to": v} for (k, v) in mapping_pairs[:100]]
            return out

        edit_res = exec_edit_sql(
            sql_steps=sql_steps,
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            target_db_path=args.get("target_db_path"),
            dest_db_name=args.get("dest_db_name"),
            preview_sql=args.get("preview_sql"),
            limit_rows=limit,
        )
        edit_res["llm_map"] = {
            "plan_id": pid,
            "fingerprint": fingerprint,
            "model": model_name,
            "mode": mode,
            "key_strategy": key_strategy,
            "write_mode": write_mode,
            "output_type": output_type,
            "options_sql_used": bool(options_sql),
            "cache": {"enabled": bool(cache_enabled), "scope": cache_scope, "cache_table": cache_table},
            "resume_offset": int(resume_offset),
            "next_offset": int(resume_offset + len(to_process)),
            "done": bool(resume_offset + len(to_process) >= len(missing_items)),
            "n_items": int(len(missing_items)),
            "n_processed": int(len(to_process)),
            "n_mapped": int(len(mapping_pairs) + len(mapping_pairs_by_row)),
            "token_usage": token_usage_total or None,
            "token_usage_reports": int(token_usage_reports),
            "failures": failures[:50],
        }
        # Return the mapping for transparency/debugging (capped).
        edit_res["mapping_preview"] = [{"from": k, "to": v} for (k, v) in mapping_pairs[:100]]
        return edit_res

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_llm_map_tool(), handler_factory=_llm_map_handler_factory)




