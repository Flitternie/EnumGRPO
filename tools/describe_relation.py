from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.db import attach_extra_required_dbs, nullsafe_str, parse_schema_table, run_readonly_query_on_conn
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime


def _validate_relation_name(name: Any) -> Tuple[Optional[str], str]:
    # This tool uses information_schema with string-literal equality checks,
    # so we can safely accept relation names that require quoting (e.g. spaces).
    return parse_schema_table(str(name or ""))


def _describe_relation_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "describe_relation",
        "description": (
            "Describe a table/view: columns, types, nullability, and basic key hints. "
            "Use this after list_relations to understand join keys and detect suspicious columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relation": {"type": "string", "description": "Relation name: table or schema.table"},
                "db_path": {
                    "type": ["string", "null"],
                    "description": "Optional override DB path for non-session usage (must be under db_files_dir or db_edit_dir).",
                },
            },
            "required": ["relation"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _describe_relation_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        schema, table = _validate_relation_name(args.get("relation"))
        db_path = nullsafe_str(args.get("db_path"))

        table_esc = table.replace("'", "''")
        where = f"table_name = '{table_esc}'"
        if schema:
            schema_esc = schema.replace("'", "''")
            where += f" AND table_schema = '{schema_esc}'"
        else:
            where += " AND table_schema NOT IN ('information_schema','pg_catalog')"

        sql_cols = (
            "SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position "
            "FROM information_schema.columns "
            f"WHERE {where} "
            "ORDER BY ordinal_position"
        )
        # DuckDB: PRAGMA table_info is convenient for PK hints, but doesn't carry schema reliably.
        # We provide PK hints best-effort by trying PRAGMA if schema is absent.
        pragma_sql: Optional[str] = None
        if schema is None:
            pragma_sql = f"PRAGMA table_info('{table_esc}')"

        def _merge(out_cols: Dict[str, Any], out_pk: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            result: Dict[str, Any] = {"columns": out_cols}
            if out_pk and isinstance(out_pk.get("rows"), list):
                # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                pk_cols: List[str] = []
                for r in out_pk.get("rows") or []:
                    if isinstance(r, list) and len(r) >= 6 and r[5]:
                        if r[1] is not None:
                            pk_cols.append(str(r[1]))
                if pk_cols:
                    result["primary_key_hint"] = pk_cols
            return result

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                )
                out_cols = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql_cols, limit_rows=2000)
                out_cols["db_files"] = [os.path.basename(str(getattr(sess, "db_path", "") or ""))]  # type: ignore[index]
                out_pk = None
                if pragma_sql:
                    try:
                        out_pk = run_readonly_query_on_conn(conn=sess.conn, sql_text=pragma_sql, limit_rows=2000)
                    except Exception:
                        out_pk = None
                merged = _merge(out_cols, out_pk)
                merged["relation"] = f"{schema + '.' if schema else ''}{table}"
                return merged

        out_cols = exec_readonly_sql(
            sql_text=sql_cols,
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            db_path=db_path,
            limit_rows=2000,
        )
        out_pk = None
        if pragma_sql:
            try:
                out_pk = exec_readonly_sql(
                    sql_text=pragma_sql,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    db_path=db_path,
                    limit_rows=2000,
                )
            except Exception:
                out_pk = None
        merged = _merge(out_cols, out_pk)
        merged["relation"] = f"{schema + '.' if schema else ''}{table}"
        return merged

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_describe_relation_tool(), handler_factory=_describe_relation_handler_factory)

