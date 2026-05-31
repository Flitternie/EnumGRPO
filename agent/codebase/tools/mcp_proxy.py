from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import ConfigDict, Field, create_model

from openhands.sdk.llm import TextContent, ImageContent
from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool import ToolDefinition, ToolExecutor

import mcp.types as mcp_types

from codebase.mcp_client import McpStdioClient


# ---------------------------------------------------------------------------
# Module-level cache for action models.
# build_action_model is called inside the global tool factory on every
# resolve_tool() call.  Without caching, each call creates a new Pydantic
# class with the same name, which Pydantic rejects as a duplicate.
# ---------------------------------------------------------------------------
_ACTION_MODEL_CACHE: dict[tuple[str, str], Type[Action]] = {}
_ACTION_MODEL_CACHE_LOCK = threading.Lock()


def _jsonschema_type_to_py(t: Any) -> Tuple[type, bool]:
    """
    Returns (python_type, is_optional).
    We intentionally keep this permissive; nested objects/arrays become dict/list.
    """
    is_optional = False
    if isinstance(t, list):
        tt = [x for x in t if x != "null"]
        if len(tt) != len(t):
            is_optional = True
        t = tt[0] if tt else "object"

    if t == "string":
        return str, is_optional
    if t == "integer":
        return int, is_optional
    if t == "number":
        return float, is_optional
    if t == "boolean":
        return bool, is_optional
    if t == "array":
        return list, is_optional
    if t == "object":
        return dict, is_optional
    return Any, True


def build_action_model(tool: mcp_types.Tool, namespace: str = "") -> Type[Action]:
    """Build a pydantic Action subclass for *tool*.

    *namespace* must be unique per ``DbAgentRuntime`` instance (e.g. a short
    hash of the run directory) so that concurrent rollouts do not register
    duplicate class names in pydantic's global discriminator registry.

    Results are cached by ``(namespace, tool_name)`` so calling this function
    multiple times (e.g. on every ``resolve_tool`` invocation) never creates
    the same class twice.
    """
    tool_name = str(tool.name or "").strip()
    cache_key = (namespace, tool_name)
    with _ACTION_MODEL_CACHE_LOCK:
        if cache_key in _ACTION_MODEL_CACHE:
            return _ACTION_MODEL_CACHE[cache_key]

        model = _build_action_model(tool, namespace)
        _ACTION_MODEL_CACHE[cache_key] = model
        return model


def _build_action_model(tool: mcp_types.Tool, namespace: str) -> Type[Action]:
    schema = tool.inputSchema or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else None
    required_set = set(required) if isinstance(required, list) else set()

    # Avoid pydantic warnings / attribute shadowing with Action's own fields.
    reserved = set(dir(Action)) | {"schema", "kind"}

    fields: Dict[str, Tuple[Any, Any]] = {}
    tool_name = str(tool.name or "").strip()
    for name, prop in props.items():
        mcp_name = name
        alias_name = name
        prop_schema = prop if isinstance(prop, dict) else {}
        py_t, opt = _jsonschema_type_to_py(prop_schema.get("type"))

        desc = str(prop_schema.get("description") or "")
        # We auto-inject session_id at runtime, so do not require it in the model schema.
        # Also relax open_session's many "required" fields so the agent can omit nulls.
        if mcp_name == "session_id" or tool_name == "open_session":
            default = None
            opt = True
        else:
            default_is_required = mcp_name in required_set
            if default_is_required and not opt:
                default = ...
            else:
                default = None
        field_name = f"{alias_name}_" if alias_name in reserved else alias_name
        ann = Optional[py_t] if (opt or default is None) else py_t
        fields[field_name] = (
            ann,
            Field(default=default, description=desc, alias=alias_name),
        )

    model_name = f"Mcp_{tool.name}_Action" if not namespace else f"{namespace}_Mcp_{tool.name}_Action"
    # NOTE: we must allow population by field name as well as alias.
    return create_model(
        model_name,
        __base__=Action,
        __config__=ConfigDict(populate_by_name=True, extra="forbid", frozen=True),
        **fields,
    )  # type: ignore[arg-type]


class McpToolObservation(Observation):
    ok: bool = True
    is_error: bool = False
    tool: str = ""
    result_text: str = ""
    error: str = ""

    @property
    def to_llm_content(self) -> List[TextContent | ImageContent]:
        if self.error:
            return [TextContent(text=f"{self.tool} ERROR: {self.error}")]
        if not self.result_text:
            return [TextContent(text=f"{self.tool}: (no output)")]
        return [TextContent(text=self.result_text)]


class McpToolExecutor(ToolExecutor[Action, McpToolObservation]):
    def __init__(
        self,
        client: McpStdioClient,
        tool_name: str,
        *,
        required_keys: List[str] | None = None,
        session_id_getter: Callable[[], Optional[str]] | None = None,
        session_id_invalidator: Callable[[], None] | None = None,
        allowed_db_path_getter: Callable[[], Optional[str]] | None = None,
    ):
        self._client = client
        self._tool_name = tool_name
        self._required_keys = list(required_keys or [])
        self._session_id_invalidator = session_id_invalidator
        self._session_id_getter = session_id_getter
        self._allowed_db_path_getter = allowed_db_path_getter
        # Cache plan arguments by plan_id for tools that require plan->run consistency.
        # This prevents plan_id mismatch when the LLM slightly changes inputs between
        # action="plan" and action="run".
        self._plan_args_by_id: Dict[str, Dict[str, Any]] = {}

    def __call__(self, action: Action, conversation=None) -> McpToolObservation:
        try:
            args = action.model_dump(exclude_none=True, by_alias=True)
            # OpenHands may include internal discriminator keys; MCP schemas reject extras.
            args.pop("kind", None)

            # MCP tool schemas often mark nullable fields as "required".
            # Ensure every required key is present, defaulting to null.
            for k in self._required_keys:
                if k not in args:
                    args[k] = None
            # Inject or enforce session_id / db_path policy.
            sid = (self._session_id_getter() if self._session_id_getter else None) or None
            allowed_db_path = (self._allowed_db_path_getter() if self._allowed_db_path_getter else None) or None

            if self._tool_name == "open_session":
                # If a live session already exists, return it directly.
                if sid:
                    payload = {"ok": True, "session_id": sid, "note": "session already open"}
                    return McpToolObservation(ok=True, is_error=False, tool=self._tool_name, result_text=json.dumps(payload, ensure_ascii=False))
                if allowed_db_path:
                    # If caller supplied db_path, enforce it; otherwise inject.
                    db_path_arg = str(args.get("db_path") or "").strip()
                    if db_path_arg and db_path_arg != allowed_db_path:
                        raise ValueError("open_session db_path must match the user-provided db_path")
                    args["db_path"] = allowed_db_path
                # Ensure required keys exist as nulls if omitted (server schema requires them).
                args.setdefault("db_name", None)
                args.setdefault("db_root", None)
                args.setdefault("db_files_dir", None)
                args.setdefault("db_edit_dir", None)
                args.setdefault("read_only", True)
            else:
                # Intercept close_session: clear the cached session_id so the next
                # open_session call actually re-opens rather than returning the dead session.
                if self._tool_name == "close_session" and self._session_id_invalidator:
                    self._session_id_invalidator()
                # Only inject session_id for tools that accept it. `list_sessions` rejects extras.
                if sid and self._tool_name != "list_sessions":
                    # Override any model-provided session_id to keep the agent pinned to the user's session.
                    args["session_id"] = sid

            # Enforce plan->run input stability for LLM operators.
            plan_snapshot: Dict[str, Any] | None = None
            if self._tool_name in {"llm_map", "llm_reduce"}:
                op = str(args.get("action") or "").strip().lower()
                if op == "plan":
                    # Snapshot what we actually send to MCP (minus session_id),
                    # then associate it with the returned plan_id after the call.
                    plan_snapshot = dict(args)
                    plan_snapshot.pop("session_id", None)
                elif op == "run":
                    pid = str(args.get("plan_id") or "").strip()
                    if pid and pid in self._plan_args_by_id:
                        cached = dict(self._plan_args_by_id[pid])
                        # Always use the cached plan args, then apply run-specific fields.
                        cached["action"] = "run"
                        cached["plan_id"] = pid
                        cached["confirm"] = bool(args.get("confirm", True))
                        # Re-attach the current session_id (runtime-injected).
                        if "session_id" in args:
                            cached["session_id"] = args["session_id"]
                        args = cached

            res = self._client.call_tool(self._tool_name, args)
            text = ""
            payload: Any = None
            if res.structuredContent is not None:
                try:
                    payload = res.structuredContent
                    text = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    payload = res.structuredContent
                    text = str(payload)
            else:
                parts: List[str] = []
                for c in (res.content or []):
                    if getattr(c, "type", None) == "text":
                        parts.append(str(getattr(c, "text", "") or ""))
                    else:
                        parts.append(str(c))
                text = "\n".join(p for p in parts if p)

                # Best-effort: parse JSON payload from text for plan caching.
                if text:
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = None

            # If this was a plan call, record plan_id -> args snapshot.
            if (
                plan_snapshot is not None
                and isinstance(payload, dict)
                and str(payload.get("action") or "").strip().lower() == "plan"
            ):
                pid = str(payload.get("plan_id") or "").strip()
                if pid:
                    self._plan_args_by_id[pid] = dict(plan_snapshot)

            return McpToolObservation(ok=not bool(res.isError), is_error=bool(res.isError), tool=self._tool_name, result_text=text)
        except Exception as e:
            return McpToolObservation(ok=False, is_error=True, tool=self._tool_name, error=str(e))


class McpProxyTool(ToolDefinition[Action, McpToolObservation]):
    """
    A generic ToolDefinition instance created per MCP tool.

    Note: the OpenHands-visible "tool name" is the name passed to register_tool(),
    not this class name.
    """

    @classmethod
    def create(cls, conv_state, **kwargs):  # type: ignore[no-untyped-def]
        # This tool is instantiated directly inside register_tool factories.
        # OpenHands marks ToolDefinition.create as abstract, so we implement it
        # only to satisfy the interface.
        raise NotImplementedError("McpProxyTool.create is not used; tools are constructed via register_tool factories.")

