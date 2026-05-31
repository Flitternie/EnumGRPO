from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from utils.db import attach_extra_required_dbs, is_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime


def _explain_sql_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "explain_sql",
        "description": (
            "Explain a read-only SELECT/WITH query without executing it (or with analyze=true, execute with profiling). "
            "Use this to catch runaway scans, cross joins, or expensive plans before running full queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Read-only SELECT/WITH query to explain."},
                "analyze": {
                    "type": ["boolean", "null"],
                    "description": "If true, run EXPLAIN ANALYZE (executes the query). Default false.",
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _explain_sql_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        sql_raw = str(args.get("sql") or "").strip()
        if not sql_raw:
            raise ValueError("sql is required")
        if not is_readonly_sql(sql_raw):
            raise ValueError("Only read-only single-statement SQL starting with SELECT or WITH is allowed.")
        sql = sql_raw.rstrip().rstrip(";").strip()

        analyze_raw = args.get("analyze")
        analyze = bool(analyze_raw) if analyze_raw is not None else False

        if getattr(rt, "session_manager", None) is None or not getattr(rt, "session_id", None):
            raise ValueError("explain_sql requires session_id (open_session first).")
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]

        stmt = ("EXPLAIN ANALYZE " if analyze else "EXPLAIN ") + sql

        with sess.lock:
            attach_extra_required_dbs(
                conn=sess.conn,
                required_dbs=list(rt.required_dbs),
                db_files_dir=str(rt.db_files_dir or ""),
            )
            cur = sess.conn.execute(stmt)
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])] if getattr(cur, "description", None) else []

        # DuckDB often returns a single column with the plan text.
        plan_text: Optional[str] = None
        if rows and isinstance(rows[0], (list, tuple)) and rows[0]:
            try:
                plan_text = str(rows[0][0])
            except Exception:
                plan_text = None

        return {
            "db_files": [os.path.basename(str(getattr(sess, "db_path", "") or ""))],
            "analyze": bool(analyze),
            "statement": stmt,
            "columns": [str(c) for c in cols],
            "rows": [list(r) for r in rows],
            "plan_text": plan_text,
        }

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_explain_sql_tool(), handler_factory=_explain_sql_handler_factory)

