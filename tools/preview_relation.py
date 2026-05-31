from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from utils.db import attach_extra_required_dbs, nullsafe_str, quote_ident, quote_qualified_name, run_readonly_query_on_conn
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime


def _quote_relation(rel: str) -> str:
    return quote_qualified_name(rel, max_parts=2)


def _validate_where(where: Any) -> Optional[str]:
    if where is None:
        return None
    if not isinstance(where, str):
        raise ValueError("where must be a string or null")
    s = where.strip()
    if not s:
        return None
    if ";" in s:
        raise ValueError("where must not contain ';'")
    if re.search(r"\b(select|update|delete|insert|alter|drop|create|attach|detach|copy|pragma|load|install)\b", s, re.I):
        raise ValueError("where contains a forbidden keyword")
    return s


def _validate_columns(cols: Any) -> Optional[List[str]]:
    if cols is None:
        return None
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError("columns must be a list of strings or null")
    out: List[str] = []
    seen = set()
    for c in cols:
        s = c.strip()
        if not s or s in seen:
            continue
        # Validate by attempting to parse/quote. This accepts identifiers with spaces
        # (and also accepts already-quoted identifiers).
        quote_ident(s)
        out.append(s)
        seen.add(s)
    return out or None


def _preview_relation_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "preview_relation",
        "description": (
            "Preview example rows from a table/view to disambiguate ambiguous column names and spot corruption. "
            "Prefer this over extending describe_relation with row examples to keep tools orthogonal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relation": {"type": "string", "description": "Relation name: table or schema.table"},
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional subset of columns to select (defaults to *).",
                },
                "where": {"type": ["string", "null"], "description": "Optional WHERE expression (single expression; no semicolons)."},
                "limit_rows": {"type": ["integer", "null"], "description": "Max rows to return (1-1000). Default 20."},
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


def _preview_relation_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        rel_sql = _quote_relation(args.get("relation"))
        cols = _validate_columns(args.get("columns"))
        where = _validate_where(args.get("where"))

        limit_raw = args.get("limit_rows")
        limit = int(limit_raw) if limit_raw is not None else 20
        limit = max(1, min(limit, 1000))

        db_path = nullsafe_str(args.get("db_path"))

        select_list = "*" if cols is None else ", ".join(quote_ident(c) for c in cols)
        sql = f"SELECT {select_list} FROM {rel_sql}"
        if where:
            sql += f" WHERE {where}"

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


TOOL_PLUGIN = ToolPlugin(tool=_preview_relation_tool(), handler_factory=_preview_relation_handler_factory)

