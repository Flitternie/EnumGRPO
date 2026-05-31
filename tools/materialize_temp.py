from __future__ import annotations

import os
from typing import Any, Callable, Dict

from utils.db import attach_extra_required_dbs, is_readonly_sql, quote_ident, run_readonly_query_on_conn
from tools.tool_base import ToolPlugin, ToolRuntime


def _schema_for_relation(*, conn: Any, name: str) -> Dict[str, Any]:
    """
    Return a schema description using information_schema (SELECT-only), so it works
    with our read-only SQL guard (which blocks PRAGMA).
    """
    name_esc = str(name or "").replace("'", "''")
    base_sql = (
        "SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position "
        "FROM information_schema.columns "
        f"WHERE table_name = '{name_esc}' AND table_schema NOT IN ('information_schema','pg_catalog') "
        "ORDER BY CASE WHEN table_schema = 'temp' THEN 0 ELSE 1 END, ordinal_position"
    )
    return run_readonly_query_on_conn(conn=conn, sql_text=base_sql, limit_rows=2000)


def _materialize_temp_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "materialize_temp",
        "description": (
            "Materialize an intermediate result as a TEMP table/view in the current session. "
            "Use this to break complex analysis into inspectable steps without editing the underlying DB."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "TEMP table/view name (identifier)."},
                "object_type": {
                    "type": "string",
                    "enum": ["table", "view"],
                    "description": "Materialize as TEMP TABLE or TEMP VIEW.",
                },
                "sql": {"type": "string", "description": "Read-only SELECT/WITH query to materialize."},
                "replace": {
                    "type": ["boolean", "null"],
                    "description": "If true, CREATE OR REPLACE. Default true.",
                },
                "row_count": {
                    "type": ["boolean", "null"],
                    "description": "If true, compute COUNT(*) for the materialized object. Default true.",
                },
            },
            "required": ["name", "object_type", "sql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _materialize_temp_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        name_raw = str(args.get("name") or "").strip()
        if not name_raw:
            raise ValueError("name is required")
        name_q = quote_ident(name_raw)

        object_type = str(args.get("object_type") or "").strip().lower()
        if object_type not in ("table", "view"):
            raise ValueError("object_type must be 'table' or 'view'")

        sql_raw = str(args.get("sql") or "").strip()
        if not sql_raw:
            raise ValueError("sql is required")
        if not is_readonly_sql(sql_raw):
            raise ValueError("Only read-only single-statement SQL starting with SELECT or WITH is allowed.")
        sql = sql_raw.rstrip().rstrip(";").strip()

        replace_raw = args.get("replace")
        replace = True if replace_raw is None else bool(replace_raw)

        row_count_raw = args.get("row_count")
        want_row_count = True if row_count_raw is None else bool(row_count_raw)

        if getattr(rt, "session_manager", None) is None or not getattr(rt, "session_id", None):
            raise ValueError("materialize_temp requires session_id (open_session first).")
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]

        verb = "CREATE OR REPLACE" if replace else "CREATE"
        stmt = f"{verb} TEMP {object_type.upper()} {name_q} AS {sql};"

        with sess.lock:
            attach_extra_required_dbs(
                conn=sess.conn,
                required_dbs=list(rt.required_dbs),
                db_files_dir=str(rt.db_files_dir or ""),
            )
            sess.conn.execute(stmt)
            schema = _schema_for_relation(conn=sess.conn, name=name_raw)
            rc = None
            if want_row_count and object_type == "table":
                out = run_readonly_query_on_conn(conn=sess.conn, sql_text=f"SELECT COUNT(*) AS row_count FROM {name_q}", limit_rows=1)
                try:
                    rc = int(out.get("rows")[0][0])  # type: ignore[index]
                except Exception:
                    rc = None

        return {
            "ok": True,
            "db_files": [os.path.basename(str(getattr(sess, "db_path", "") or ""))],
            "name": name_raw,
            "object_type": object_type,
            "statement": stmt,
            "row_count": rc,
            "schema": schema,
        }

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_materialize_temp_tool(), handler_factory=_materialize_temp_handler_factory)

