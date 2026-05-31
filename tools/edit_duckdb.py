from __future__ import annotations

import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import (
    attach_extra_required_dbs,
    append_edits_log,
    exec_sql_steps_on_conn,
    is_readonly_sql,
    resolve_duckdb_path,
    run_readonly_query_on_conn,
    split_sql_statements,
)


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _json_safe(v: Any) -> Any:
    return None if _is_nan(v) else v


def _normalize_path_under(dir_abs: str, p: str) -> str:
    if not isinstance(p, str) or not p.strip():
        raise ValueError("Empty path.")
    s = p.strip()
    if os.path.isabs(s):
        out = os.path.abspath(s)
    else:
        out = os.path.abspath(os.path.join(dir_abs, s))
    if out == dir_abs or out.startswith(dir_abs + os.sep):
        return out
    raise ValueError("Path must stay within db_edit_dir.")


def _clone_db_to_edited(*, src_path: str, edited_dir: str, dest_db_name: Optional[str]) -> str:
    os.makedirs(edited_dir, exist_ok=True)
    base = (dest_db_name or "").strip() or os.path.basename(src_path)
    if not base.lower().endswith(".duckdb"):
        base = f"{base}.duckdb"
    dest = _normalize_path_under(edited_dir, base)

    if os.path.exists(dest):
        stem, _ext = os.path.splitext(os.path.basename(dest))
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = _normalize_path_under(edited_dir, f"{stem}__copy_{ts}.duckdb")

    # copy2 preserves metadata/permissions; many source DBs are read-only, so ensure the clone is writable.
    shutil.copy2(src_path, dest)
    try:
        st_mode = os.stat(dest).st_mode
        os.chmod(dest, st_mode | 0o200)
    except Exception:
        pass
    return dest


def exec_edit_sql(
    *,
    sql_steps: List[Any],
    required_dbs: List[str],
    db_files_dir: str,
    db_edit_dir: str,
    target_db_path: Optional[str],
    dest_db_name: Optional[str],
    preview_sql: Optional[str],
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

    primary_src = resolve_duckdb_path(db_dir, required_dbs[0])
    if not primary_src:
        raise RuntimeError(f"Primary DuckDB file not found for {required_dbs[0]} in {db_dir}")

    # Precedence: explicit arg (ToolRuntime/CLI) > env override > tool default.
    edited_dir_abs = os.path.abspath(db_edit_dir or os.getenv("SANDBOX_DIR") or str(Path(__file__).resolve().parent.parent / "sandbox"))
    os.makedirs(edited_dir_abs, exist_ok=True)

    cloned_from: Optional[str] = None
    if isinstance(target_db_path, str) and target_db_path.strip():
        edited_db_path = _normalize_path_under(edited_dir_abs, target_db_path)
        if not os.path.isfile(edited_db_path):
            raise FileNotFoundError(f"target_db_path does not exist: {edited_db_path}")
    else:
        edited_db_path = _clone_db_to_edited(src_path=primary_src, edited_dir=edited_dir_abs, dest_db_name=dest_db_name)
        cloned_from = primary_src

    if not isinstance(sql_steps, list) or not sql_steps:
        raise ValueError("sql_steps must be a non-empty list of strings.")

    statements: List[str] = []
    for step in sql_steps:
        if not isinstance(step, str) or not step.strip():
            continue
        statements.extend(split_sql_statements(step))
    statements = [s.strip() for s in statements if isinstance(s, str) and s.strip()]
    if not statements:
        raise ValueError("No executable SQL statements found in sql_steps.")

    sql_log = "\n\n".join(s.rstrip(";").strip() + ";" for s in statements)

    limit_rows = int(limit_rows or 0)
    if limit_rows <= 0:
        limit_rows = 200
    limit_rows = max(1, min(limit_rows, 1000))

    conn = None
    attached: List[str] = []
    try:
        conn = duckdb.connect(edited_db_path, read_only=False)

        for extra in required_dbs[1:]:
            extra_path = resolve_duckdb_path(db_dir, extra)
            if not extra_path:
                raise RuntimeError(f"Extra DuckDB file not found for {extra} in {db_dir}")
            alias = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(extra_path))[0]) or "db"
            safe_path = extra_path.replace("'", "''")
            conn.execute(f"ATTACH DATABASE '{safe_path}' AS {alias} (READ_ONLY);")
            attached.append(os.path.basename(extra_path))

        conn.execute("BEGIN TRANSACTION;")
        try:
            for stmt in statements:
                conn.execute(stmt)
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise

        # Best-effort: append a combined SQL log next to the edited DB.
        try:
            stem = os.path.splitext(os.path.basename(edited_db_path))[0]
            log_path = _normalize_path_under(edited_dir_abs, f"{stem}__edits.sql")
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"-- edits at {ts}\n")
                f.write(sql_log)
                f.write("\n\n")
        except Exception:
            pass

        preview: Optional[Dict[str, Any]] = None
        if isinstance(preview_sql, str) and preview_sql.strip():
            if not is_readonly_sql(preview_sql):
                raise ValueError("preview_sql must be read-only SELECT/WITH.")
            q = preview_sql.strip()
            if q.endswith(";"):
                q = q[:-1].strip()
            wrapped = f"SELECT * FROM ({q}) AS q LIMIT {limit_rows}"
            df = conn.execute(wrapped).fetchdf()
            cols = [str(c) for c in list(df.columns)]
            rows = [[_json_safe(v) for v in r] for r in df.values.tolist()]
            preview = {"columns": cols, "rows": rows, "row_count": int(len(rows)), "limit_rows": int(limit_rows)}

        return {
            "ok": True,
            "edited_db_path": edited_db_path,
            "cloned_from": cloned_from,
            "attached_db_files": attached,
            "sql_log": sql_log,
            "statements_executed": statements,
            "preview": preview,
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _edit_duckdb_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "edit_duckdb",
        "description": (
            "Create or continue editing an editable copy of the current DuckDB database. "
            "If target_db_path is not provided, this will clone the primary required DuckDB into db_edit_dir "
            "and apply the given SQL steps (DDL/DML). If target_db_path is provided, it must point to an existing "
            "DB under db_edit_dir and edits will be applied to that DB. "
            "All SQL steps are executed in order and logged together."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "List of SQL statements/steps to apply (may include DDL/DML). "
                        "If a step contains multiple statements separated by semicolons, they will be split and executed in order."
                    ),
                },
                "target_db_path": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional path (absolute or relative to db_edit_dir) to an existing editable DB under db_edit_dir "
                        "to continue editing across rounds. If null, a fresh editable copy will be created by cloning the "
                        "primary required DuckDB."
                    ),
                },
                "dest_db_name": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional file name for the cloned DB to create under db_edit_dir. "
                        "Ignored if target_db_path is provided."
                    ),
                },
                "preview_sql": {
                    "type": ["string", "null"],
                    "description": "Optional read-only SELECT/WITH to run after edits for a small preview.",
                },
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max rows to return for preview_sql (1-1000). If null, defaults to 200.",
                },
            },
            "required": ["sql_steps"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _edit_duckdb_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        limit_rows = args.get("limit_rows")
        if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
            sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
            # For read-only sessions (e.g. default chat session), fall back to the normal edit_duckdb behavior
            # which clones into db_edit_dir and applies edits there. We also reuse a single working copy
            # for the rest of the session to avoid creating many copies/logs.
            if getattr(sess, "read_only", False):
                # Read working_db_path under the lock so concurrent first-edit calls don't
                # both see None and each clone a separate copy.
                with sess.lock:
                    working = getattr(sess, "working_db_path", None)

                if isinstance(working, str) and working.strip():
                    args = dict(args)
                    args["target_db_path"] = working
                    args["dest_db_name"] = None
                else:
                    # First write in this session: clone once with a stable session-scoped name.
                    base = (list(rt.required_dbs)[0] if rt.required_dbs else "DB").strip() if isinstance(rt.required_dbs, (list, tuple)) else "DB"
                    sid8 = str(rt.session_id or "")[:8]
                    args = dict(args)
                    if not (isinstance(args.get("dest_db_name"), str) and str(args.get("dest_db_name")).strip()):
                        args["dest_db_name"] = f"{base}__chat_{sid8}.duckdb"
                    args["target_db_path"] = None
                res = exec_edit_sql(
                    sql_steps=args.get("sql_steps") or [],
                    required_dbs=list(rt.required_dbs),
                    db_files_dir=str(rt.db_files_dir or ""),
                    db_edit_dir=str(rt.db_edit_dir or ""),
                    target_db_path=args.get("target_db_path"),
                    dest_db_name=args.get("dest_db_name"),
                    preview_sql=args.get("preview_sql"),
                    limit_rows=int(limit_rows) if limit_rows is not None else 200,
                )
                edited_path = res.get("edited_db_path")
                if isinstance(edited_path, str) and edited_path.strip():
                    # Write back under lock; double-check in case a concurrent call already set it.
                    with sess.lock:
                        if not getattr(sess, "working_db_path", None):
                            try:
                                sess.working_db_path = edited_path
                            except Exception:
                                pass
                    # Promote session to point at the working copy so subsequent run_sql sees edits.
                    try:
                        rt.session_manager.rebind(rt.session_id, db_path=edited_path, read_only=False)  # type: ignore[union-attr]
                    except Exception:
                        pass
                res["note"] = (res.get("note") or "") + " (chat session working copy: reused across edits)"
                return res

            if not getattr(sess, "read_only", False):
                # If caller provided target_db_path, require it to match session db_path (avoid surprises).
                tdp = args.get("target_db_path")
                if isinstance(tdp, str) and tdp.strip():
                    want = os.path.abspath(tdp.strip())
                    have = os.path.abspath(str(getattr(sess, "db_path", "") or ""))
                    if want != have:
                        raise ValueError("target_db_path must match the session's db_path when using session_id.")
                with sess.lock:
                    attach_extra_required_dbs(
                        conn=sess.conn,
                        required_dbs=list(rt.required_dbs),
                        db_files_dir=str(rt.db_files_dir or ""),
                        read_only=True,
                    )
                    sql_log, statements = exec_sql_steps_on_conn(conn=sess.conn, sql_steps=args.get("sql_steps") or [])
                    append_edits_log(
                        db_edit_dir=str(rt.db_edit_dir or ""),
                        edited_db_path=str(getattr(sess, "db_path", "") or ""),
                        sql_log=sql_log,
                    )
                    preview_sql = args.get("preview_sql")
                    preview = None
                    if isinstance(preview_sql, str) and preview_sql.strip():
                        if not is_readonly_sql(preview_sql):
                            raise ValueError("preview_sql must be read-only SELECT/WITH.")
                        preview = run_readonly_query_on_conn(
                            conn=sess.conn,
                            sql_text=preview_sql,
                            limit_rows=int(limit_rows) if limit_rows is not None else 200,
                        )
                    return {
                        "ok": True,
                        "edited_db_path": str(getattr(sess, "db_path", "") or ""),
                        "cloned_from": None,
                        "attached_db_files": [],
                        "sql_log": sql_log,
                        "statements_executed": statements,
                        "preview": preview,
                        "note": "Session mode: edits were applied directly on the existing session connection.",
                    }
        return exec_edit_sql(
            sql_steps=args.get("sql_steps") or [],
            required_dbs=list(rt.required_dbs),
            db_files_dir=str(rt.db_files_dir or ""),
            db_edit_dir=str(rt.db_edit_dir or ""),
            target_db_path=args.get("target_db_path"),
            dest_db_name=args.get("dest_db_name"),
            preview_sql=args.get("preview_sql"),
            limit_rows=int(limit_rows) if limit_rows is not None else 200,
        )

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_edit_duckdb_tool(), handler_factory=_edit_duckdb_handler_factory)

