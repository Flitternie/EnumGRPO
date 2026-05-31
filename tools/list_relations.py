from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from utils.db import attach_extra_required_dbs, nullsafe_str, run_readonly_query_on_conn
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime


def _list_relations_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_relations",
        "description": (
            "List available tables and views to resolve ambiguous schemas. "
            "Use this early to discover candidate relations before writing joins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "schema_name": {"type": ["string", "null"], "description": "Optional schema name filter."},
                "like": {"type": ["string", "null"], "description": "Optional substring filter on table/view name."},
                "include_system": {
                    "type": ["boolean", "null"],
                    "description": "If true, include system schemas (information_schema, pg_catalog). Default false.",
                },
                "limit": {"type": ["integer", "null"], "description": "Max relations to return (1-2000). Default 500."},
                "db_path": {
                    "type": ["string", "null"],
                    "description": "Optional override DB path for non-session usage (must be under db_files_dir or db_edit_dir).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _list_relations_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        schema_name = nullsafe_str(args.get("schema_name"))
        like = nullsafe_str(args.get("like"))

        include_system_raw = args.get("include_system")
        include_system = bool(include_system_raw) if include_system_raw is not None else False

        limit_raw = args.get("limit")
        limit = int(limit_raw) if limit_raw is not None else 500
        limit = max(1, min(limit, 2000))

        db_path = nullsafe_str(args.get("db_path"))

        where_parts: List[str] = []
        if not include_system:
            where_parts.append("table_schema NOT IN ('information_schema','pg_catalog')")
        if schema_name:
            schema_esc = schema_name.replace("'", "''")
            where_parts.append(f"table_schema = '{schema_esc}'")
        if like:
            s = like.replace("'", "''").lower()
            where_parts.append(f"lower(table_name) LIKE '%{s}%'")

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = (
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables"
            f"{where_sql} "
            "ORDER BY table_schema, table_name "
            f"LIMIT {int(limit)}"
        )

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                )
                out = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql, limit_rows=limit)
                out["db_files"] = [os.path.basename(str(getattr(sess, "db_path", "") or ""))]  # type: ignore[index]
                return out

        return exec_readonly_sql(
            sql_text=sql,
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            db_path=db_path,
            limit_rows=limit,
        )

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_list_relations_tool(), handler_factory=_list_relations_handler_factory)

