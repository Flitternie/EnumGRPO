from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------
# MCP stdio transport uses stdout exclusively for JSON-RPC messages.
# Any non-JSON bytes on stdout corrupt the protocol and produce
# "Failed to parse JSONRPC message" errors on the client side.
#
# Replace sys.stdout with a thin wrapper that forwards all print()/write()
# calls to stderr, while keeping the original sys.stdout object (and its
# .buffer) accessible so mcp.server.stdio can still write JSON-RPC frames.
# -----------------------------------------------------------------------
class _StdoutGuard:
    """Intercept text-level writes and redirect them to stderr.

    mcp.server.stdio writes JSON-RPC via sys.stdout.buffer (the raw binary
    stream), so we expose the real stdout's .buffer unchanged.  Only the
    text wrapper (print, logging handlers) is redirected.
    """

    def __init__(self, real_stdout, real_stderr):
        self._real = real_stdout
        self._err = real_stderr
        # Expose the raw binary buffer -- mcp.server.stdio uses this.
        self.buffer = real_stdout.buffer

    def write(self, s: str) -> int:
        return self._err.write(s)

    def flush(self) -> None:
        self._err.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return self._err.encoding

    def fileno(self) -> int:
        return self._real.fileno()

    # Delegate anything else (e.g. .errors, .name) to stderr.
    def __getattr__(self, name: str):
        return getattr(self._err, name)


sys.stdout = _StdoutGuard(sys.__stdout__, sys.__stderr__)

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.server import request_ctx
from mcp.server.models import InitializationOptions

# Ensure workspace root is on sys.path for local imports.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from tools import get_tool_definitions, get_tool_handlers  # noqa: E402
from tools.tool_base import ToolRuntime  # noqa: E402
from tools.run_sql import make_run_sql_plugin as _make_run_sql_plugin  # noqa: E402
from tools.run_blendsql import make_run_blendsql_plugin as _make_run_blendsql_plugin  # noqa: E402

# When MCP_TEXT2SQL_AGENT=1, the server exposes only run_sql (unrestricted SQL)
_TEXT2SQL: bool = os.getenv("MCP_TEXT2SQL_AGENT", "").strip().lower() in ("1", "true", "yes")
_TEXT2SQL_PLUGIN = _make_run_sql_plugin(strict_readonly=False)

# When MCP_BLENDSQL_AGENT=1, expose run_sql (unrestricted) + run_blendsql.
_BLENDSQL_AGENT: bool = os.getenv("MCP_BLENDSQL_AGENT", "").strip().lower() in ("1", "true", "yes")
_BLENDSQL_PLUGIN = _make_run_blendsql_plugin()

# When MCP_NOSCHEMTOOLS_AGENT=1, expose full tool set minus the dedicated schema
# inspection tools (list_relations / describe_relation / preview_relation), and
# use unrestricted run_sql so the agent can query information_schema freely.
_NOSCHEMTOOLS: bool = os.getenv("MCP_NOSCHEMTOOLS_AGENT", "").strip().lower() in ("1", "true", "yes")
_NOSCHEMTOOLS_SQL_PLUGIN = _make_run_sql_plugin(strict_readonly=False)


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _default_db_files_dir() -> str:
    return _abs_path(os.getenv("DB_FILES_DIR"))


def _default_db_edit_dir() -> str:
    # Tool/MCP default: keep editable DBs under sandbox unless overridden.
    return _abs_path(os.getenv("SANDBOX_DIR") or str(_ROOT / "sandbox"))


def _resolve_db_path(*, db_path: Optional[str], db_name: Optional[str], db_root: Optional[str], db_files_dir: str) -> str:
    if isinstance(db_path, str) and db_path.strip():
        p = db_path.strip()
        return _abs_path(p) if os.path.isabs(p) else _abs_path(str(_ROOT / p))

    name = (db_name or "").strip()
    if not name:
        raise ValueError("open_session requires either db_path or db_name")
    if not name.lower().endswith(".duckdb"):
        name = f"{name}.duckdb"
    root = _abs_path(db_root) if isinstance(db_root, str) and db_root.strip() else db_files_dir
    return _abs_path(os.path.join(root, name))


def _is_under(dir_abs: str, p_abs: str) -> bool:
    d = os.path.abspath(dir_abs)
    p = os.path.abspath(p_abs)
    return p == d or p.startswith(d + os.sep)


@dataclass
class _Session:
    session_id: str
    conn: Any
    db_path: str
    read_only: bool
    lock: threading.RLock
    # ToolRuntime inputs:
    required_dbs: List[str]
    db_files_dir: str
    db_edit_dir: str
    # Used by edit_duckdb's "working copy" pattern.
    working_db_path: Optional[str] = None


class SessionManager:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._sessions: Dict[str, _Session] = {}

    def open(
        self,
        *,
        db_path: Optional[str],
        db_name: Optional[str],
        db_root: Optional[str],
        db_files_dir: Optional[str],
        db_edit_dir: Optional[str],
        read_only: bool,
    ) -> Dict[str, Any]:
        try:
            import duckdb  # type: ignore
        except Exception as e:
            raise RuntimeError("duckdb python package is not installed") from e

        files_dir = _abs_path(db_files_dir) if isinstance(db_files_dir, str) and db_files_dir.strip() else _default_db_files_dir()
        edited_dir = _abs_path(db_edit_dir) if isinstance(db_edit_dir, str) and db_edit_dir.strip() else _default_db_edit_dir()
        os.makedirs(edited_dir, exist_ok=True)

        primary = _resolve_db_path(db_path=db_path, db_name=db_name, db_root=db_root, db_files_dir=files_dir)
        if not os.path.isfile(primary):
            raise FileNotFoundError(f"db not found: {primary}")

        # Enforce safety: writable sessions must use a DB under edited_dir.
        if not read_only and not _is_under(edited_dir, primary):
            raise ValueError("read_only=false requires db_path under db_edit_dir")

        sid = secrets.token_hex(16)
        conn = duckdb.connect(primary, read_only=bool(read_only))
        sess = _Session(
            session_id=sid,
            conn=conn,
            db_path=primary,
            read_only=bool(read_only),
            lock=threading.RLock(),
            required_dbs=[Path(primary).stem.upper()],
            db_files_dir=files_dir,
            db_edit_dir=edited_dir,
        )
        with self._mu:
            self._sessions[sid] = sess
        return {
            "ok": True,
            "session_id": sid,
            "db_path": primary,
            "read_only": bool(read_only),
            "db_files_dir": files_dir,
            "db_edit_dir": edited_dir,
            "required_dbs": list(sess.required_dbs),
        }

    def close(self, session_id: str) -> Dict[str, Any]:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        with self._mu:
            sess = self._sessions.pop(sid, None)
        if sess is None:
            return {"ok": True, "closed": False, "session_id": sid}
        try:
            with sess.lock:
                sess.conn.close()
        except Exception:
            pass
        return {"ok": True, "closed": True, "session_id": sid}

    def get(self, session_id: str) -> _Session:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        with self._mu:
            sess = self._sessions.get(sid)
        if sess is None:
            raise ValueError(f"Unknown session_id: {sid}")
        return sess

    def rebind(self, session_id: str, *, db_path: str, read_only: bool) -> Dict[str, Any]:
        """
        Rebind session to a different DB file. Note: this will create a new DuckDB connection
        and therefore drops TEMP objects.
        """
        try:
            import duckdb  # type: ignore
        except Exception as e:
            raise RuntimeError("duckdb python package is not installed") from e
        sess = self.get(session_id)
        new_path = _abs_path(db_path) if os.path.isabs(db_path) else _abs_path(str(_ROOT / db_path))
        if not os.path.isfile(new_path):
            raise FileNotFoundError(f"db_path not found: {new_path}")
        if not read_only and not _is_under(sess.db_edit_dir, new_path):
            raise ValueError("read_only=false requires db_path under db_edit_dir")
        with sess.lock:
            try:
                sess.conn.close()
            except Exception:
                pass
            sess.conn = duckdb.connect(new_path, read_only=bool(read_only))
            sess.db_path = new_path
            sess.read_only = bool(read_only)
            sess.required_dbs = [Path(new_path).stem.upper()]
        return {"ok": True, "session_id": sess.session_id, "db_path": sess.db_path, "read_only": sess.read_only}

    def set_roots(
        self,
        session_id: str,
        *,
        db_files_dir: Optional[str],
        db_edit_dir: Optional[str],
    ) -> Dict[str, Any]:
        sess = self.get(session_id)
        files_dir = _abs_path(db_files_dir) if isinstance(db_files_dir, str) and db_files_dir.strip() else sess.db_files_dir
        edited_dir = _abs_path(db_edit_dir) if isinstance(db_edit_dir, str) and db_edit_dir.strip() else sess.db_edit_dir
        os.makedirs(edited_dir, exist_ok=True)
        with sess.lock:
            sess.db_files_dir = files_dir
            sess.db_edit_dir = edited_dir
        return {
            "ok": True,
            "session_id": sess.session_id,
            "db_files_dir": sess.db_files_dir,
            "db_edit_dir": sess.db_edit_dir,
        }

    def list(self) -> Dict[str, Any]:
        with self._mu:
            items = list(self._sessions.values())
        out: List[Dict[str, Any]] = []
        for s in items:
            out.append(
                {
                    "session_id": s.session_id,
                    "db_path": s.db_path,
                    "read_only": bool(s.read_only),
                    "db_files_dir": s.db_files_dir,
                    "db_edit_dir": s.db_edit_dir,
                    "required_dbs": list(s.required_dbs),
                }
            )
        return {"ok": True, "sessions": out}


_SESSIONS = SessionManager()
_SERVER = Server("db-revise-tools")

def _resolve_logs_dir() -> Path:
    """
    Where to write MCP session logs.

    Default: repo-root `./logs/` (historical behavior).
    When invoked by the agent runtime, it can pass RUNNING_DIR so logs are
    written under that run's log directory instead of a shared root folder.
    """
    run_dir = (os.getenv("RUNNING_DIR") or "").strip()
    if run_dir:
        try:
            p = Path(run_dir)
            run_abs = (Path(_ROOT) / p).resolve() if not p.is_absolute() else p.resolve()
            return run_abs / "logs" / "mcp_server"
        except Exception:
            pass
    return _ROOT / "logs"


_LOGS_DIR = _resolve_logs_dir()
_LOG_LOCK = threading.Lock()
_SESSION_LOG_PATHS: Dict[str, Path] = {}


def _safe_preview(v: Any, *, max_chars: int = 8_000) -> Any:
    """
    Best-effort: keep logs readable and JSON-serializable.
    """
    try:
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            return v if len(v) <= max_chars else (v[:max_chars] + "…(truncated)")
        if isinstance(v, (list, tuple)):
            out = [_safe_preview(x, max_chars=max_chars) for x in v[:200]]
            if len(v) > 200:
                out.append(f"…({len(v) - 200} more items)")
            return out
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for i, (k, vv) in enumerate(v.items()):
                if i >= 200:
                    out["…"] = f"({len(v) - 200} more keys)"
                    break
                out[str(k)] = _safe_preview(vv, max_chars=max_chars)
            return out
        return str(v)
    except Exception:
        return "<unserializable>"


def _extract_llm_token_usage_from_result(result: Any) -> Dict[str, Any]:
    """
    Pull token usage out of tool results for top-level logging.

    Expected shapes:
    - llm_reduce: {"token_usage": {...}}
    - llm_map: {"llm_map": {"token_usage": {...}, "token_usage_reports": N}}
    """
    usage: Optional[Dict[str, int]] = None
    reports: Optional[int] = None
    try:
        if not isinstance(result, dict):
            return {}

        tu = result.get("token_usage")
        if isinstance(tu, dict) and tu:
            usage = {}
            for k, v in tu.items():
                try:
                    usage[str(k)] = int(v)  # type: ignore[arg-type]
                except Exception:
                    pass

        lm = result.get("llm_map")
        if isinstance(lm, dict):
            lm_tu = lm.get("token_usage")
            if isinstance(lm_tu, dict) and lm_tu:
                usage = {}
                for k, v in lm_tu.items():
                    try:
                        usage[str(k)] = int(v)  # type: ignore[arg-type]
                    except Exception:
                        pass
            lm_reports = lm.get("token_usage_reports")
            if lm_reports is not None:
                try:
                    reports = int(lm_reports)  # type: ignore[arg-type]
                except Exception:
                    reports = None
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    if usage:
        out["llm_token_usage"] = usage
    if reports is not None:
        out["llm_token_usage_reports"] = reports
    return out


def _session_log_path(session_id: str) -> Path:
    sid = str(session_id or "").strip()
    if sid in _SESSION_LOG_PATHS:
        return _SESSION_LOG_PATHS[sid]
    safe = "".join(c for c in sid if (c.isalnum() or c in ("-", "_"))) or "unknown"
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = _LOGS_DIR / f"session_{ts}_{safe}.jsonl"
    _SESSION_LOG_PATHS[sid] = p
    return p


def _append_session_log(session_id: str, event: Dict[str, Any]) -> None:
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        payload.setdefault("session_id", session_id)
        line = json.dumps(_safe_preview(payload), ensure_ascii=False, separators=(",", ":"))
        with _LOG_LOCK:
            with _session_log_path(session_id).open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Never break tool execution due to logging failures.
        return


def _tool_from_plugin_schema(tool: Dict[str, Any]) -> types.Tool:
    # ToolPlugin schemas follow OpenAI function format: {"type":"function","name","description","parameters",...}
    name = str(tool.get("name") or "")
    desc = str(tool.get("description") or "")
    raw_schema = tool.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(raw_schema, dict):
        raw_schema = {"type": "object", "properties": {}}

    # MCP server runtime requires session_id for all plugin tool calls.
    # We inject it into the exposed input schema, but remove it before invoking handlers.
    input_schema = copy.deepcopy(raw_schema)
    if str(input_schema.get("type") or "") != "object":
        input_schema = {"type": "object", "properties": {}}
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    if "session_id" not in props:
        props = dict(props)
        props["session_id"] = {
            "type": "string",
            "description": "Session id from open_session (required for all tool calls except open_session/list_sessions).",
        }
    input_schema["properties"] = props
    req = input_schema.get("required")
    if not isinstance(req, list):
        req = []
    if "session_id" not in req:
        req = list(req) + ["session_id"]
    input_schema["required"] = req

    return types.Tool(name=name, description=desc, inputSchema=input_schema)


def _text_result(payload: Any, *, tool_name: str = "") -> List[types.TextContent]:
    try:
        # Tool outputs must be serializable; use a safe fallback so a single
        # unserializable value doesn't turn a successful call into an error.
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = json.dumps({"error": "serialization failed"}, ensure_ascii=False)
    text = _maybe_truncate_result(text, tool_name=tool_name)
    return [types.TextContent(type="text", text=text)]


# Tools whose results can balloon the context with raw row data.
_DATA_TOOLS = frozenset({"run_sql", "preview_relation", "describe_relation", "list_relations"})

# Maximum chars for a single tool result sent to the agent.
# Override with env var MCP_RESULT_MAX_CHARS (0 = disabled).
def _get_result_max_chars() -> int:
    raw = os.getenv("MCP_RESULT_MAX_CHARS", "").strip()
    try:
        v = int(raw)
        return v if v > 0 else 0
    except (ValueError, TypeError):
        return 4_000  # default: ~1K tokens


def _maybe_truncate_result(text: str, *, tool_name: str = "") -> str:
    """
    For data-returning tools, trim the JSON result to at most MCP_RESULT_MAX_CHARS
    by dropping trailing rows and appending a truncation note.

    We do row-level trimming (not a raw string cut) so the agent always receives
    valid JSON and a clear signal about how many rows were dropped.
    """
    if tool_name not in _DATA_TOOLS:
        return text
    max_chars = _get_result_max_chars()
    if max_chars == 0 or len(text) <= max_chars:
        return text

    # Try row-level trimming on the parsed object.
    try:
        obj = json.loads(text)
    except Exception:
        # Not JSON — fall back to hard string truncation.
        return text[:max_chars] + f'…(truncated, original length {len(text)})'

    rows = obj.get("rows")
    if not isinstance(rows, list) or not rows:
        # No rows to drop; hard truncation of the serialised form.
        truncated = text[:max_chars]
        return truncated + f'…(truncated, original length {len(text)})'

    total_rows = len(rows)
    kept = list(rows)
    while kept:
        obj["rows"] = kept
        obj["row_count"] = len(kept)
        candidate = json.dumps(obj, ensure_ascii=False, default=str)
        if len(candidate) <= max_chars:
            if len(kept) < total_rows:
                obj["truncated"] = f"Showing {len(kept)}/{total_rows} rows (result truncated to fit context)."
            return json.dumps(obj, ensure_ascii=False, default=str)
        # Drop the last quarter of kept rows each iteration for O(log n) passes.
        drop = max(1, len(kept) // 4)
        kept = kept[: len(kept) - drop]

    # Even zero rows is too large — hard truncate.
    return text[:max_chars] + f'…(truncated, original length {len(text)})'


@_SERVER.list_tools()
async def list_tools() -> List[types.Tool]:
    tools: List[types.Tool] = []
    # Session management tools
    tools.extend(
        [
            types.Tool(
                name="open_session",
                description="Open a DuckDB session (persistent connection for TEMP tables). Call this first; pass the returned session_id to all subsequent tool calls. Default read_only=true unless you need to write data. Do not call close_session until the task is complete.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": ["string", "null"], "description": "Path to .duckdb file (absolute or relative to workspace)."},
                        "db_name": {"type": ["string", "null"], "description": "DB filename stem or name; resolved under db_root or db_files_dir."},
                        "db_root": {"type": ["string", "null"], "description": "Directory to resolve db_name (defaults to db_files_dir)."},
                        "db_files_dir": {"type": ["string", "null"], "description": "Base dir for read-only DBs (used for resolving/attaching extras)."},
                        "db_edit_dir": {"type": ["string", "null"], "description": "Base dir for writable DBs + edit logs."},
                        "read_only": {"type": "boolean", "description": "If true, open read-only; can still use TEMP intermediates."},
                    },
                    "required": ["db_path", "db_name", "db_root", "db_files_dir", "db_edit_dir", "read_only"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="close_session",
                description="Close a previously opened session.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="list_sessions",
                description="List open sessions.",
                inputSchema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            ),
            types.Tool(
                name="set_session_roots",
                description="Update db_files_dir / db_edit_dir for an existing session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "db_files_dir": {"type": ["string", "null"]},
                        "db_edit_dir": {"type": ["string", "null"]},
                    },
                    "required": ["session_id", "db_files_dir", "db_edit_dir"],
                    "additionalProperties": False,
                },
            ),
        ]
    )

    # BlendSQL agent mode: expose run_blendsql only alongside session tools.
    if _BLENDSQL_AGENT:
        tools.append(_tool_from_plugin_schema(_BLENDSQL_PLUGIN.tool))
        return tools

    # Ablation mode: only expose run_sql (unrestricted) alongside session tools.
    if _TEXT2SQL:
        tools.append(_tool_from_plugin_schema(_TEXT2SQL_PLUGIN.tool))
        return tools

    # No-schema-tools mode: full tool set with unrestricted run_sql replacing
    # the strict variant. Tool filtering is handled at the agent side.
    if _NOSCHEMTOOLS:
        for t in get_tool_definitions():
            if not (isinstance(t, dict) and t.get("type") == "function"):
                continue
            if str(t.get("name") or "") == "run_sql":
                tools.append(_tool_from_plugin_schema(_NOSCHEMTOOLS_SQL_PLUGIN.tool))
            else:
                tools.append(_tool_from_plugin_schema(t))
        return tools

    # Full mode: expose all tool plugins.
    for t in get_tool_definitions():
        if isinstance(t, dict) and t.get("type") == "function":
            tools.append(_tool_from_plugin_schema(t))
    return tools


@_SERVER.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    tool_name = str(name or "").strip()
    args = arguments if isinstance(arguments, dict) else {}

    if tool_name == "open_session":
        start = time.time()
        out = _SESSIONS.open(
            db_path=args.get("db_path"),
            db_name=args.get("db_name"),
            db_root=args.get("db_root"),
            db_files_dir=args.get("db_files_dir"),
            db_edit_dir=args.get("db_edit_dir"),
            read_only=bool(args.get("read_only")),
        )
        sid = str(out.get("session_id") or "").strip()
        if sid:
            _append_session_log(
                sid,
                {
                    "tool": tool_name,
                    "arguments": dict(args),
                    "duration_ms": int((time.time() - start) * 1000),
                    "result": out,
                },
            )
        return _text_result(out)
    if tool_name == "close_session":
        start = time.time()
        sid = str(args.get("session_id") or "").strip()
        out = _SESSIONS.close(sid)
        if sid:
            _append_session_log(
                sid,
                {
                    "tool": tool_name,
                    "arguments": {"session_id": sid},
                    "duration_ms": int((time.time() - start) * 1000),
                    "result": out,
                },
            )
        return _text_result(out)
    if tool_name == "list_sessions":
        return _text_result(_SESSIONS.list())
    if tool_name == "set_session_roots":
        start = time.time()
        sid = str(args.get("session_id") or "").strip()
        out = _SESSIONS.set_roots(
            sid,
            db_files_dir=args.get("db_files_dir"),
            db_edit_dir=args.get("db_edit_dir"),
        )
        if sid:
            _append_session_log(
                sid,
                {
                    "tool": tool_name,
                    "arguments": dict(args),
                    "duration_ms": int((time.time() - start) * 1000),
                    "result": out,
                },
            )
        return _text_result(out)

    # Dispatch to existing tool plugins. These tools expect to run within a ToolRuntime.
    session_id = str(args.get("session_id") or "").strip()
    if not session_id:
        _append_session_log(
            "missing_session_id",
            {
                "tool": tool_name,
                "arguments": _safe_preview(dict(args)),
                "duration_ms": 0,
                "error": "Missing session_id",
            },
        )
        return _text_result({"error": "Missing session_id. Call open_session first and pass session_id to tool calls."})

    try:
        sess = _SESSIONS.get(session_id)
    except Exception as e:
        _append_session_log(
            session_id,
            {
                "tool": tool_name,
                "arguments": _safe_preview(dict(args)),
                "duration_ms": 0,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        return _text_result({"error": str(e)})
    # Build a progress callback if the client supplied a progressToken.
    progress_callback = None
    try:
        ctx = request_ctx.get()
        progress_token = ctx.meta.progressToken if ctx.meta else None
        if progress_token is not None:
            loop = asyncio.get_running_loop()
            session = ctx.session

            def _progress_cb(progress: float, total: Optional[float], message: Optional[str]) -> None:
                asyncio.run_coroutine_threadsafe(
                    session.send_progress_notification(progress_token, progress, total=total, message=message),
                    loop,
                )

            progress_callback = _progress_cb
    except LookupError:
        pass

    rt = ToolRuntime(
        required_dbs=list(sess.required_dbs),
        db_files_dir=str(sess.db_files_dir or ""),
        db_edit_dir=str(sess.db_edit_dir or ""),
        session_manager=_SESSIONS,
        session_id=session_id,
        progress_callback=progress_callback,
    )
    if _BLENDSQL_AGENT:
        handlers = {
            _BLENDSQL_PLUGIN.tool["name"]: _BLENDSQL_PLUGIN.handler_factory(rt),
        }
    elif _TEXT2SQL:
        handlers = {_TEXT2SQL_PLUGIN.tool["name"]: _TEXT2SQL_PLUGIN.handler_factory(rt)}
    elif _NOSCHEMTOOLS:
        handlers = get_tool_handlers(rt)
        handlers[_NOSCHEMTOOLS_SQL_PLUGIN.tool["name"]] = _NOSCHEMTOOLS_SQL_PLUGIN.handler_factory(rt)
    else:
        handlers = get_tool_handlers(rt)
    h = handlers.get(tool_name)
    if h is None:
        _append_session_log(
            session_id,
            {
                "tool": tool_name,
                "arguments": _safe_preview(dict(args)),
                "duration_ms": 0,
                "error": f"Unknown tool: {tool_name}",
            },
        )
        return _text_result({"error": f"Unknown tool: {tool_name}"})

    # Remove session_id before passing args into strict tools (they don't accept it).
    tool_args = dict(args)
    tool_args.pop("session_id", None)
    start = time.time()
    try:
        out = await asyncio.to_thread(h, tool_args)
    except Exception as e:
        out = {"error": str(e)}
        err_event: Dict[str, Any] = {
            "tool": tool_name,
            "arguments": dict(tool_args),
            "duration_ms": int((time.time() - start) * 1000),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        partial_tu = getattr(e, "partial_token_usage", None)
        if isinstance(partial_tu, dict):
            err_event["partial_token_usage"] = partial_tu
            partial_calls = getattr(e, "partial_num_generation_calls", None)
            if partial_calls is not None:
                err_event["partial_num_generation_calls"] = int(partial_calls)
        _append_session_log(session_id, err_event)
    else:
        event = {
            "tool": tool_name,
            "arguments": dict(tool_args),
            "duration_ms": int((time.time() - start) * 1000),
            "result": out,
        }
        try:
            event.update(_extract_llm_token_usage_from_result(out))
        except Exception:
            pass
        _append_session_log(session_id, event)
    return _text_result(out, tool_name=tool_name)


async def _run() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await _SERVER.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="db-revise-tools",
                server_version="0.1.0",
                capabilities=_SERVER.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

