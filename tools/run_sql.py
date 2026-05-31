from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import (
    attach_extra_required_dbs,
    is_readonly_sql,
    resolve_duckdb_path,
    run_readonly_query_on_conn,
    strip_sql_comments,
    _check_no_external_access,
)


def _exec_any_sql_on_conn(*, conn: Any, sql_text: str, limit_rows: int = 200) -> Dict[str, Any]:
    """Execute any SQL on an existing connection (no readonly guardrail).

    For SELECT/WITH, applies the standard LIMIT wrapping and returns rows.
    For DDL/DML, executes directly and returns a status dict (or rows if the
    driver produces a result set, e.g. RETURNING clauses).
    """
    s = strip_sql_comments(sql_text).strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    if not s:
        raise ValueError("sql must be a non-empty string")

    _check_no_external_access(s)

    limit_rows = max(1, min(int(limit_rows or 200), 1000))

    if is_readonly_sql(s):
        wrapped = f"SELECT * FROM ({s}) AS q LIMIT {limit_rows}"
        df = conn.execute(wrapped).fetchdf()
        cols = [str(c) for c in list(df.columns)]
        rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
        return {"columns": cols, "rows": rows, "row_count": int(len(rows)), "limit_rows": limit_rows}

    # DDL / DML: execute directly; try to surface any returned rows.
    result = conn.execute(s)
    try:
        df = result.fetchdf()
        cols = [str(c) for c in list(df.columns)]
        rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
        return {"columns": cols, "rows": rows, "row_count": int(len(rows))}
    except Exception:
        return {"ok": True, "message": "Statement executed successfully."}


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _json_safe(v: Any) -> Any:
    return None if _is_nan(v) else v


def _is_single_temp_ddl(sql_text: str) -> bool:
    """
    Allow a single TEMP DDL statement (for session mode only).

    DuckDB permits CREATE TEMP TABLE on read_only connections because it doesn't modify the DB file,
    but it *does* require a persistent connection/session (TEMP objects are connection-scoped).
    """
    s = strip_sql_comments(sql_text).strip()
    if not s:
        return False
    # Allow a single trailing semicolon, but not multiple statements.
    if ";" in s:
        s2 = s.rstrip()
        if s2.endswith(";"):
            s2 = s2[:-1]
        if ";" in s2:
            return False
        s = s2.strip()
    return bool(re.match(r"(?is)^\s*create\s+(temp|temporary)\s+(table|view)\b", s))


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_under_dir(dir_abs: Path, p_abs: Path) -> bool:
    d = str(dir_abs)
    p = str(p_abs)
    return p == d or p.startswith(d + os.sep)


def _resolve_output_path(output_path: str) -> Path:
    root = _workspace_root()
    p = (output_path or "").strip()
    if not p:
        raise ValueError("output_path must be a non-empty string")
    out = Path(p)
    out_abs = (root / out).resolve() if not out.is_absolute() else out.resolve()
    if not _is_under_dir(root.resolve(), out_abs):
        raise ValueError("output_path must be under the workspace root")
    if out_abs.suffix.lower() != ".csv":
        raise ValueError("output_path must end with .csv")
    return out_abs


def _write_rows_to_csv(*, path: Path, columns: List[str], rows: List[List[Any]], include_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if include_header:
            w.writerow([str(c) for c in columns])
        for r in rows:
            if not isinstance(r, list):
                continue
            w.writerow(["" if v is None else str(v) for v in r])


def exec_readonly_sql(
    *,
    sql_text: str,
    required_dbs: List[str],
    db_files_dir: str,
    db_edit_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    limit_rows: int = 200,
) -> Dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except Exception as e:
        raise RuntimeError("duckdb python package is not installed") from e

    db_dir = db_files_dir
    if not os.path.isdir(db_dir):
        raise RuntimeError(f"DuckDB files dir not found: {db_dir}")
    if not required_dbs:
        raise ValueError("This SQL does not declare required DuckDB file(s) in its header.")
    if not is_readonly_sql(sql_text):
        raise ValueError(
            "Only read-only single-statement SQL starting with SELECT or WITH is allowed. "
            "Do not use PRAGMA (e.g., PRAGMA table_info) for schema inspection; use list_relations/describe_relation instead."
        )

    limit_rows = int(limit_rows or 0)
    if limit_rows <= 0:
        limit_rows = 200
    limit_rows = max(1, min(limit_rows, 1000))

    primary_path: Optional[str] = None
    if isinstance(db_path, str) and db_path.strip():
        p = db_path.strip()
        primary_path = os.path.abspath(p) if os.path.isabs(p) else os.path.abspath(os.path.join(db_dir, p))
        if not os.path.isfile(primary_path):
            raise FileNotFoundError(f"db_path not found: {primary_path}")
        files_dir_abs = os.path.abspath(db_dir)
        # Precedence: explicit arg > env override > tool default.
        edited_dir_abs = os.path.abspath(
            db_edit_dir
            if isinstance(db_edit_dir, str) and db_edit_dir
            else (os.getenv("SANDBOX_DIR") or str(_workspace_root() / "sandbox"))
        )
        allowed = False
        if primary_path == files_dir_abs or primary_path.startswith(files_dir_abs + os.sep):
            allowed = True
        if edited_dir_abs and (primary_path == edited_dir_abs or primary_path.startswith(edited_dir_abs + os.sep)):
            allowed = True
        if not allowed:
            raise ValueError("db_path must be under db_files_dir or db_edit_dir.")
    else:
        primary_path = resolve_duckdb_path(db_dir, required_dbs[0])
        if not primary_path:
            raise RuntimeError(f"Primary DuckDB file not found for {required_dbs[0]} in {db_dir}")

    conn = None
    try:
        conn = duckdb.connect(primary_path, read_only=True)
        attached: List[str] = [os.path.basename(primary_path)]
        for extra in required_dbs[1:]:
            extra_path = resolve_duckdb_path(db_dir, extra)
            if not extra_path:
                raise RuntimeError(f"Extra DuckDB file not found for {extra} in {db_dir}")
            alias = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(extra_path))[0]) or "db"
            safe_path = extra_path.replace("'", "''")
            conn.execute(f"ATTACH DATABASE '{safe_path}' AS {alias};")
            attached.append(os.path.basename(extra_path))

        # Enforce a hard row limit without trusting the caller to include LIMIT.
        q = (sql_text or "").strip()
        if q.endswith(";"):
            q = q[:-1].strip()
        wrapped = f"SELECT * FROM ({q}) AS q LIMIT {limit_rows}"
        df = conn.execute(wrapped).fetchdf()

        cols = [str(c) for c in list(df.columns)]
        rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
        return {
            "db_files": attached,
            "columns": cols,
            "rows": rows,
            "row_count": int(len(rows)),
            "limit_rows": int(limit_rows),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _run_sql_tool(*, strict_readonly: bool = True) -> Dict[str, Any]:
    _no_external = (
        "Do NOT use glob() or ATTACH — querying files outside the provided database is not allowed."
    )
    if strict_readonly:
        description = (
            "Execute a read-only SQL query that must start with SELECT or WITH. "
            "Use this to validate query logic and inspect sample output rows. "
            "Do NOT use this for schema inspection (no PRAGMA table_info / SHOW / DESCRIBE). "
            + _no_external
        )
        sql_description = (
            "SQL to execute. Must be a single statement starting with SELECT or WITH. "
            "Do not use for schema inspection (no PRAGMA/SHOW/DESCRIBE). "
            "glob() and ATTACH are not allowed."
        )
    else:
        description = (
            "Execute a SQL statement against the database. "
            "Supports SELECT, CREATE TABLE/VIEW, INSERT, UPDATE, DELETE, DROP, and other SQL. "
            "For SELECT/WITH queries, results are returned up to limit_rows. "
            "For DDL/DML, the statement is executed and a status is returned. "
            + _no_external
        )
        sql_description = (
            "SQL statement to execute. Any valid DuckDB SQL is accepted: "
            "SELECT, CREATE TEMP TABLE, INSERT, UPDATE, DELETE, DROP, etc. "
            "glob() and ATTACH are not allowed."
        )
    return {
        "type": "function",
        "name": "run_sql",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": sql_description,
                },
                "db_path": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional override DB path (must be under db_files_dir or db_edit_dir). "
                        "If null, uses the primary DB declared in the SQL header."
                    ),
                },
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max rows to return (1-1000). If null, defaults to 200.",
                },
                "output_path": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional .csv path to write results to (relative to workspace root, or absolute within it). "
                        "If null, results are not written. Must end with .csv."
                    ),
                },
                "include_header": {
                    "type": ["boolean", "null"],
                    "description": "For CSV output: whether to include a header row. Default true.",
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _run_sql_handler_factory(rt: ToolRuntime, *, strict_readonly: bool = True) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        sql_text = args.get("sql")
        _dp = args.get("db_path")
        db_path = (_dp if not isinstance(_dp, str) else (_dp.strip() if _dp.strip().lower() != "null" else None))
        limit_rows = args.get("limit_rows")
        output_path_raw = args.get("output_path")
        _op = str(output_path_raw).strip() if isinstance(output_path_raw, str) else ""
        output_path = _op if _op and _op.lower() != "null" else None
        include_header_raw = args.get("include_header")
        include_header = True if include_header_raw is None else bool(include_header_raw)
        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            with sess.lock:
                attach_extra_required_dbs(conn=sess.conn, required_dbs=list(rt.required_dbs), db_files_dir=str(rt.db_files_dir or ""))
                s = str(sql_text or "")
                if strict_readonly:
                    if _is_single_temp_ddl(s):
                        raise ValueError("run_sql is read-only. Use materialize_temp to create TEMP tables/views.")
                    out = run_readonly_query_on_conn(
                        conn=sess.conn,
                        sql_text=s,
                        limit_rows=int(limit_rows) if limit_rows is not None else 200,
                    )
                else:
                    out = _exec_any_sql_on_conn(
                        conn=sess.conn,
                        sql_text=s,
                        limit_rows=int(limit_rows) if limit_rows is not None else 200,
                    )
                out["db_files"] = [os.path.basename(str(getattr(sess, "db_path", "") or ""))]  # type: ignore[index]
                if output_path:
                    out_path = _resolve_output_path(output_path)
                    _write_rows_to_csv(
                        path=out_path,
                        columns=[str(c) for c in (out.get("columns") or [])],
                        rows=[r for r in (out.get("rows") or []) if isinstance(r, list)],
                        include_header=bool(include_header),
                    )
                    out["saved"] = {"path": str(out_path), "format": "csv"}  # type: ignore[index]
                return out
        out = exec_readonly_sql(
            sql_text=str(sql_text or ""),
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            db_path=str(db_path) if isinstance(db_path, str) and db_path.strip() else None,
            limit_rows=int(limit_rows) if limit_rows is not None else 200,
        )
        if output_path:
            out_path = _resolve_output_path(output_path)
            _write_rows_to_csv(
                path=out_path,
                columns=[str(c) for c in (out.get("columns") or [])],
                rows=[r for r in (out.get("rows") or []) if isinstance(r, list)],
                include_header=bool(include_header),
            )
            out["saved"] = {"path": str(out_path), "format": "csv"}  # type: ignore[index]
        return out

    return handler


def make_run_sql_plugin(*, strict_readonly: bool = True) -> ToolPlugin:
    """Return a run_sql ToolPlugin.

    Args:
        strict_readonly: When True (default, full version), only SELECT/WITH SQL is
            accepted and validated before execution.  When False (ablation version),
            any DuckDB SQL is accepted; access control is delegated to the DuckDB
            connection's own read_only setting.
    """
    return ToolPlugin(
        tool=_run_sql_tool(strict_readonly=strict_readonly),
        handler_factory=lambda rt: _run_sql_handler_factory(rt, strict_readonly=strict_readonly),
    )


# Default plugin used by the full agent (strict read-only guardrail).
TOOL_PLUGIN = make_run_sql_plugin(strict_readonly=True)

