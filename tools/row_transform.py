from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Tuple

from tools.edit_duckdb import exec_edit_sql
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import (
    attach_extra_required_dbs,
    append_edits_log,
    exec_sql_steps_on_conn,
    quote_ident,
    quote_table_name,
    run_readonly_query_on_conn,
    sql_literal,
    validate_type_sql,
    validate_where_clause,
)


def _validate_action(action: Any) -> str:
    a = str(action or "").strip().lower()
    if a in ("fetch", "apply"):
        return a
    raise ValueError("action must be 'fetch' or 'apply'")


def _validate_column_list(cols: Any, *, fallback: Sequence[str]) -> List[str]:
    if cols is None:
        return list(fallback)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError("columns must be a list of strings or null")
    out = [c.strip() for c in cols if isinstance(c, str) and c.strip()]
    if not out:
        out = list(fallback)
    # Validate idents (and allow duplicates; we'll dedupe while preserving order)
    seen = set()
    deduped: List[str] = []
    for c in out:
        if c in seen:
            continue
        quote_ident(c)  # validates
        seen.add(c)
        deduped.append(c)
    return deduped


def _validate_updates(updates: Any) -> List[Tuple[Any, Any]]:
    if not isinstance(updates, list) or not updates:
        raise ValueError("updates must be a non-empty list")
    if len(updates) > 5000:
        raise ValueError("updates too large; send in chunks (max 5000 per request)")
    out: List[Tuple[Any, Any]] = []
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError("updates items must be objects")
        if "key" not in item:
            raise ValueError("updates items must include 'key'")
        key = item.get("key")
        val = item.get("value")
        # key/value literals validated later (when rendering), but basic allowed types now:
        if not isinstance(key, (int, float, str, bool)) and key is not None:
            raise ValueError("update.key must be a JSON scalar (string/number/bool) or null")
        if not isinstance(val, (int, float, str, bool)) and val is not None:
            raise ValueError("update.value must be a JSON scalar (string/number/bool) or null")
        out.append((key, val))
    return out


def _chunk(seq: List[Tuple[Any, Any]], n: int) -> List[List[Tuple[Any, Any]]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def _row_transform_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "row_transform",
        "description": (
            "Helper for client-side row/column transforms. "
            "Use action='fetch' to read key + selected columns, then run any custom function on the client. "
            "Use action='apply' to write back per-key updates, optionally creating a new target column first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["fetch", "apply"]},
                "table": {"type": "string", "description": "Target table name: table or schema.table"},
                "key_column": {"type": "string", "description": "Unique key column used to match rows."},
                "db_path": {
                    "type": ["string", "null"],
                    "description": "Optional override DB path (must be under db_files_dir or db_edit_dir).",
                },
                # fetch
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "For action='fetch': additional columns to return (besides key_column).",
                },
                "where": {
                    "type": ["string", "null"],
                    "description": "Optional WHERE expression (single expression; no semicolons).",
                },
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max rows to return for fetch (1-1000). If null, defaults to 200.",
                },
                # apply
                "target_column": {
                    "type": ["string", "null"],
                    "description": "For action='apply': column to update (existing) or create (if create_column=true).",
                },
                "create_column": {
                    "type": ["boolean", "null"],
                    "description": "If true, add target_column to the table first (requires new_column_type).",
                },
                "new_column_type": {
                    "type": ["string", "null"],
                    "description": "SQL type for new column (e.g. VARCHAR, INTEGER, DECIMAL(18,2)).",
                },
                "updates": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "object",
                        "properties": {
                            # JSON scalar (or null). Matches runtime validation in _validate_updates.
                            "key": {"type": ["string", "number", "boolean", "null"]},
                            "value": {"type": ["string", "number", "boolean", "null"]},
                        },
                        "required": ["key", "value"],
                        "additionalProperties": False,
                    },
                    "description": "For action='apply': list of {key, value} pairs to write back.",
                },
                "target_db_path": {
                    "type": ["string", "null"],
                    "description": "Same as edit_duckdb: existing editable DB under db_edit_dir to continue editing.",
                },
                "dest_db_name": {
                    "type": ["string", "null"],
                    "description": "Same as edit_duckdb: optional filename for new cloned edited DB (ignored if target_db_path provided).",
                },
                "preview_sql": {
                    "type": ["string", "null"],
                    "description": "Optional read-only SELECT/WITH to preview after apply.",
                },
            },
            "required": ["action", "table", "key_column"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _row_transform_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        action = _validate_action(args.get("action"))
        table = quote_table_name(args.get("table"))
        key_col = str(args.get("key_column") or "").strip()
        key_col_q = quote_ident(key_col)
        where = validate_where_clause(args.get("where"))

        db_path = args.get("db_path")
        db_path_norm = str(db_path).strip() if isinstance(db_path, str) and db_path.strip() else None

        limit_rows = args.get("limit_rows")
        limit = int(limit_rows) if limit_rows is not None else 200

        if action == "fetch":
            # Always include key_column; then add requested columns.
            cols = _validate_column_list(args.get("columns"), fallback=[])
            if key_col not in cols:
                cols = [key_col] + cols
            select_list = ", ".join(quote_ident(c) for c in cols)
            sql = f"SELECT {select_list} FROM {table}"
            if where:
                sql += f" WHERE {where}"
            if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
                sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
                with sess.lock:
                    attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                    out = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql, limit_rows=limit)
                    out["db_files"] = [os.path.basename(str(getattr(sess, "db_path", "") or ""))]  # type: ignore[index]
                    return out
            return exec_readonly_sql(
                sql_text=sql,
                required_dbs=list(rt.required_dbs),
                db_files_dir=str(rt.db_files_dir or ""),
                db_edit_dir=str(rt.db_edit_dir or ""),
                db_path=db_path_norm,
                limit_rows=limit,
            )

        # action == "apply"
        target_col = args.get("target_column")
        if not isinstance(target_col, str) or not target_col.strip():
            raise ValueError("target_column is required for action='apply'")
        target_col_q = quote_ident(target_col.strip())

        create_col = args.get("create_column")
        create = bool(create_col) if create_col is not None else False
        type_sql = validate_type_sql(args.get("new_column_type")) if create else ""

        updates_raw = args.get("updates")
        updates = _validate_updates(updates_raw)

        # Build SQL steps: optional ALTER, then one or more UPDATE ... FROM (VALUES ...)
        sql_steps: List[str] = []
        if create:
            sql_steps.append(f"ALTER TABLE {table} ADD COLUMN {target_col_q} {type_sql};")

        # Chunk values to keep SQL size reasonable
        for chunk in _chunk(updates, 1000):
            values_rows = ",\n".join(f"({sql_literal(k)}, {sql_literal(v)})" for (k, v) in chunk)
            values_cte = f"(VALUES\n{values_rows}\n) AS u(key, value)"
            upd = f"UPDATE {table} AS t SET {target_col_q} = u.value FROM {values_cte} WHERE t.{key_col_q} = u.key"
            if where:
                upd += f" AND ({where})"
            upd += ";"
            sql_steps.append(upd)

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            if getattr(sess, "read_only", False):
                raise ValueError(
                    "row_transform action='apply' cannot write on a read-only session. "
                    "Open the session with read_only=false, or use edit_duckdb with a DB under db_edit_dir."
                )
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    read_only=True,
                )
                sql_log, statements = exec_sql_steps_on_conn(conn=sess.conn, sql_steps=sql_steps)
                append_edits_log(
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    edited_db_path=str(getattr(sess, "db_path", "") or ""),
                    sql_log=sql_log,
                )
                preview_sql = args.get("preview_sql")
                preview = None
                if isinstance(preview_sql, str) and preview_sql.strip():
                    preview = run_readonly_query_on_conn(conn=sess.conn, sql_text=preview_sql, limit_rows=limit)
                return {
                    "ok": True,
                    "edited_db_path": str(getattr(sess, "db_path", "") or ""),
                    "cloned_from": None,
                    "attached_db_files": [],
                    "sql_log": sql_log,
                    "statements_executed": statements,
                    "preview": preview,
                    "note": "Session mode: updates were applied directly on the existing session connection.",
                }

        return exec_edit_sql(
            sql_steps=sql_steps,
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            target_db_path=args.get("target_db_path"),
            dest_db_name=args.get("dest_db_name"),
            preview_sql=args.get("preview_sql"),
            limit_rows=limit,
        )

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_row_transform_tool(), handler_factory=_row_transform_handler_factory)




