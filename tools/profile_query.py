from __future__ import annotations

import datetime as _dt
import math
import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from utils.db import attach_extra_required_dbs, nullsafe_str, run_readonly_query_on_conn
from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _try_iso_dt(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (_dt.date, _dt.datetime)):
        return True
    s = str(x).strip()
    if not s:
        return False
    # ISO-ish only (keeps this dependency-free and predictable)
    try:
        _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def _truncate(v: Any, max_chars: int) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        return v
    s = str(v)
    if max_chars > 0 and len(s) > max_chars:
        return s[: max(0, max_chars - 3)] + "..."
    return s


def _profile_query_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "profile_query",
        "description": (
            "Run a small read-only query and compute lightweight column profiles in Python. "
            "Use this to detect corrupted/mixed-format columns before writing complex SQL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Read-only SELECT/WITH query to profile (should be a sample)."},
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max rows to fetch for profiling (1-1000). If null, defaults to 200.",
                },
                "max_cell_chars": {
                    "type": ["integer", "null"],
                    "description": "Max characters kept per cell for examples/top values (10-500). Default 120.",
                },
                "db_path": {
                    "type": ["string", "null"],
                    "description": "Optional override DB path for non-session usage (must be under db_files_dir or db_edit_dir).",
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _profile_rows(columns: List[str], rows: List[List[Any]], *, max_cell_chars: int) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"row_count": int(n), "columns": []}
    if not columns:
        return out

    # Transpose by column index (best-effort).
    for j, col in enumerate(columns):
        vals: List[Any] = []
        for r in rows:
            if not isinstance(r, list):
                continue
            vals.append(r[j] if j < len(r) else None)

        nulls = sum(1 for v in vals if v is None)
        non_null = [v for v in vals if v is not None]
        examples = [_truncate(v, max_cell_chars) for v in non_null[:10]]

        # Distinct/top values (stringified, truncated) for quick inspection
        sv = [str(_truncate(v, max_cell_chars)) for v in non_null]
        top = Counter(sv).most_common(10)

        floats: List[float] = []
        numeric_ok = 0
        dt_ok = 0
        for v in non_null:
            fv = _try_float(v)
            if fv is not None:
                numeric_ok += 1
                floats.append(fv)
            if _try_iso_dt(v):
                dt_ok += 1

        numeric_stats: Optional[Dict[str, Any]] = None
        if floats:
            floats_sorted = sorted(floats)
            numeric_stats = {
                "min": floats_sorted[0],
                "max": floats_sorted[-1],
                "p50": floats_sorted[len(floats_sorted) // 2],
            }

        # String length stats (on stringified non-nulls)
        lengths = [len(str(v)) for v in non_null]
        length_stats: Optional[Dict[str, Any]] = None
        if lengths:
            lengths_sorted = sorted(lengths)
            length_stats = {
                "min": int(lengths_sorted[0]),
                "max": int(lengths_sorted[-1]),
                "p50": int(lengths_sorted[len(lengths_sorted) // 2]),
            }

        out["columns"].append(
            {
                "name": str(col),
                "null_count": int(nulls),
                "null_rate": (float(nulls) / float(n)) if n else None,
                "distinct_count_sample": int(len(set(sv))),
                "top_values_sample": [{"value": v, "count": int(c)} for (v, c) in top],
                "examples": examples,
                "numeric_parse_success_rate": (float(numeric_ok) / float(len(non_null))) if non_null else None,
                "iso_datetime_parse_success_rate": (float(dt_ok) / float(len(non_null))) if non_null else None,
                "numeric_stats_sample": numeric_stats,
                "length_stats_sample": length_stats,
            }
        )

    return out


def _profile_query_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        sql_raw = str(args.get("sql") or "").strip()
        if not sql_raw:
            raise ValueError("sql is required")

        limit_rows_raw = args.get("limit_rows")
        limit_rows = int(limit_rows_raw) if limit_rows_raw is not None else 200
        limit_rows = max(1, min(limit_rows, 1000))

        max_cell_chars_raw = args.get("max_cell_chars")
        max_cell_chars = int(max_cell_chars_raw) if max_cell_chars_raw is not None else 120
        max_cell_chars = max(10, min(max_cell_chars, 500))

        db_path = nullsafe_str(args.get("db_path"))

        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(
                    conn=sess.conn,
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                )
                res = run_readonly_query_on_conn(conn=sess.conn, sql_text=sql_raw, limit_rows=limit_rows)
                res["db_files"] = [os.path.basename(str(getattr(sess, "db_path", "") or ""))]  # type: ignore[index]
        else:
            res = exec_readonly_sql(
                sql_text=sql_raw,
                required_dbs=list(rt.required_dbs),
                db_files_dir=str(rt.db_files_dir or ""),
                db_edit_dir=str(rt.db_edit_dir or ""),
                db_path=db_path,
                limit_rows=limit_rows,
            )

        cols = [str(c) for c in (res.get("columns") or [])]
        rows = [r for r in (res.get("rows") or []) if isinstance(r, list)]

        profile = _profile_rows(cols, rows, max_cell_chars=max_cell_chars)
        return {
            "ok": True,
            "query": {"limit_rows": int(limit_rows)},
            "profile": profile,
        }

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_profile_query_tool(), handler_factory=_profile_query_handler_factory)

