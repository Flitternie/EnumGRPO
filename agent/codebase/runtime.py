"""Autonomous runtime for the database agent.

This runtime exposes DuckDB/database operations exclusively via MCP tools
served by `mcp_server.py` (see `.cursor/mcp.json`), and intentionally does NOT
register file/terminal editing tools.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from openhands.sdk import Agent, LLM, LocalConversation
from openhands.sdk.context.condenser import LLMSummarizingCondenser, PipelineCondenser
from openhands.sdk.context import AgentContext
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.security.analyzer import SecurityAnalyzerBase
from openhands.sdk.security.confirmation_policy import ConfirmRisky
from openhands.sdk.security.risk import SecurityRisk
from codebase.config import MAX_ITERATION_PER_RUN, RuntimeConfig
from codebase.condenser import ObservationPruningCondenser
from codebase.mcp_client import McpServerConfig, McpStdioClient
from codebase.tools.mcp_proxy import McpProxyTool, McpToolExecutor, McpToolObservation, build_action_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from utils.llm import apply_litellm_provider_kwargs


def _set_tool_text_content_limit(limit: int) -> None:
    """Override OpenHands tool text truncation limit at runtime."""
    if limit <= 0:
        return
    try:
        import openhands.sdk.llm.message as llm_message
        llm_message.DEFAULT_TEXT_CONTENT_LIMIT = limit
    except Exception:
        pass
    try:
        import openhands.sdk.utils as sdk_utils
        sdk_utils.DEFAULT_TEXT_CONTENT_LIMIT = limit
    except Exception:
        pass
    try:
        import openhands.sdk.utils.truncate as truncate_utils
        truncate_utils.DEFAULT_TEXT_CONTENT_LIMIT = limit
    except Exception:
        pass


def _load_cursor_mcp_server(workspace_dir: Path, *, preferred_name: str | None = None) -> McpServerConfig | None:
    """
    Best-effort: read `.cursor/mcp.json` and build a server config.
    If preferred_name is provided, use that server name; otherwise pick the first.
    """
    cfg_path = workspace_dir / ".cursor" / "mcp.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        return None

    chosen = None
    if preferred_name and preferred_name in servers:
        chosen = servers.get(preferred_name)
    if chosen is None:
        # Heuristic: prefer a server mentioning our repo's mcp_server.py
        for name, s in servers.items():
            if not isinstance(s, dict):
                continue
            args = s.get("args")
            if isinstance(args, list) and any("mcp_server.py" in str(x) for x in args):
                chosen = s
                break
        if chosen is None:
            chosen = next(iter(servers.values()))

    if not isinstance(chosen, dict):
        return None

    command = str(chosen.get("command") or "").strip()
    args = chosen.get("args") if isinstance(chosen.get("args"), list) else []
    env = chosen.get("env") if isinstance(chosen.get("env"), dict) else {}
    env2: dict[str, str] = {str(k): str(v) for (k, v) in env.items()}
    return McpServerConfig(command=command, args=[str(x) for x in args], env=env2, cwd=str(workspace_dir))


def _load_mcp_server_config(workspace_dir: Path) -> McpServerConfig:
    """
    Resolution order:
    1) Environment variables DB_MCP_COMMAND/DB_MCP_ARGS_JSON/DB_MCP_ENV_JSON
    2) `.cursor/mcp.json` (prefer server name DB_MCP_SERVER_NAME if set)
    3) Fallback to `python -u mcp_server.py`
    """
    cmd = (os.getenv("DB_MCP_COMMAND") or "").strip()
    args_json = (os.getenv("DB_MCP_ARGS_JSON") or "").strip()
    env_json = (os.getenv("DB_MCP_ENV_JSON") or "").strip()
    if cmd:
        try:
            args = json.loads(args_json) if args_json else ["-u", "mcp_server.py"]
        except Exception:
            args = ["-u", "mcp_server.py"]
        if not isinstance(args, list):
            args = ["-u", "mcp_server.py"]
        try:
            env = json.loads(env_json) if env_json else {}
        except Exception:
            env = {}
        if not isinstance(env, dict):
            env = {}
        return McpServerConfig(command=cmd, args=[str(x) for x in args], env={str(k): str(v) for (k, v) in env.items()}, cwd=str(workspace_dir))

    preferred = (os.getenv("DB_MCP_SERVER_NAME") or "").strip() or None
    cursor_cfg = _load_cursor_mcp_server(workspace_dir, preferred_name=preferred)
    if cursor_cfg:
        return cursor_cfg

    return McpServerConfig(command="python", args=["-u", "mcp_server.py"], env={}, cwd=str(workspace_dir))


class _ToolNameSecurityAnalyzer(SecurityAnalyzerBase):
    """Assign HIGH risk to selected tool names so OpenHands pauses for confirmation."""

    high_risk_tools: set[str]

    def security_risk(self, action) -> SecurityRisk:  # type: ignore[override]
        tool_name = str(getattr(action, "tool_name", "") or "").strip()
        if tool_name in self.high_risk_tools:
            # Special-case LLM operators: only "run" should require confirmation.
            if tool_name in {"llm_map", "llm_reduce"}:
                try:
                    tool_action = getattr(action, "action", None)
                    payload = (
                        tool_action.model_dump(exclude_none=False, by_alias=True)
                        if tool_action is not None and hasattr(tool_action, "model_dump")
                        else {}
                    )
                except Exception:
                    payload = {}
                op = str((payload or {}).get("action") or "").strip().lower()
                return SecurityRisk.HIGH if op == "run" else SecurityRisk.LOW

            return SecurityRisk.HIGH
        if tool_name:
            return SecurityRisk.LOW
        return SecurityRisk.UNKNOWN

    def analyze_pending_actions(self, action_events):  # type: ignore[no-untyped-def]
        return [(ev, self.security_risk(ev)) for ev in (action_events or [])]


class DbAgentRuntime:
    """Runtime for the database agent (MCP tools only)."""

    def __init__(self, workspace_dir: Path, run_dir: Path, runtime_cfg, *, auto_mode: bool = False, system_prompt_path: Optional[str] = None, db_files_dir: Optional[str] = None, on_action=None, blocked_tools: Optional[list] = None):
        if isinstance(runtime_cfg, str):
            runtime_cfg = RuntimeConfig(model_name=runtime_cfg)

        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
        self.auto_mode = bool(auto_mode)
        self.system_prompt_path = system_prompt_path
        self.model_name = runtime_cfg.model_name
        self.api_key = runtime_cfg.api_key
        # Optional callback(tool_name: str, action_count: int) invoked after each tool call.
        self._on_action = on_action
        self.base_url = runtime_cfg.base_url
        self.temperature = runtime_cfg.temperature
        self.blocked_tools: set = set(blocked_tools) if blocked_tools else set()
        self.max_iteration_per_run: int = (
            runtime_cfg.max_iteration_per_run
            if isinstance(getattr(runtime_cfg, "max_iteration_per_run", None), int)
            else MAX_ITERATION_PER_RUN
        )
        self.tool_text_content_limit = 120000
        _set_tool_text_content_limit(self.tool_text_content_limit)
        self.max_output_tokens = 8192 * 2

        # Unique namespace for pydantic action model names — prevents duplicate
        # class registration errors when multiple DbAgentRuntime instances run
        # concurrently (each creates its own set of Mcp_*_Action models).
        import hashlib as _hashlib
        self._model_namespace = _hashlib.md5(str(run_dir).encode()).hexdigest()[:8]

        self.completions_dir = self.run_dir / "logs" / "completions"
        self.completions_dir.mkdir(parents=True, exist_ok=True)

        # Main agent LLM
        self.main_llm = self._build_llm(usage_id="main")

        # Collect all LLMs for metrics
        self._llms: list[LLM] = [self.main_llm]

        # MCP tool client (DuckDB tools)
        self._mcp_cfg = _load_mcp_server_config(self.workspace_dir)
        # Route MCP server logs into this run's log directory (when the agent spawns the server).
        mcp_env = dict(self._mcp_cfg.env or {})
        mcp_env.setdefault("RUNNING_DIR", str(self.run_dir.resolve()))
        # Propagate db_files_dir so the MCP server knows where to find databases.
        # This avoids requiring DB_FILES_DIR to be set in the shell environment.
        if db_files_dir:
            mcp_env.setdefault("DB_FILES_DIR", str(Path(db_files_dir).resolve()))
        if runtime_cfg.mcp_result_max_chars is not None:
            mcp_env["MCP_RESULT_MAX_CHARS"] = str(runtime_cfg.mcp_result_max_chars)
        elif os.environ.get("MCP_RESULT_MAX_CHARS"):
            mcp_env.setdefault("MCP_RESULT_MAX_CHARS", os.environ["MCP_RESULT_MAX_CHARS"])
        self._mcp_cfg = McpServerConfig(
            command=self._mcp_cfg.command,
            args=list(self._mcp_cfg.args or []),
            env=mcp_env,
            cwd=self._mcp_cfg.cwd,
        )
        self._mcp_client = McpStdioClient(self._mcp_cfg)
        self._mcp_tools: list[str] = []
        self._allowed_db_path: Optional[str] = None
        self._default_session_id: Optional[str] = None

        # Build per-instance ToolDefinition objects (mcp_client baked in).
        # These are injected into agent._tools via _pre_init_agent() in run().
        self._build_mcp_tool_definitions()

        # Build agent and conversation
        self.agent = self._build_main_agent()
        self.conversation: LocalConversation | None = None

    def set_db_context(self, *, db_path: str, read_only: bool = True) -> None:
        """
        Pin the runtime to a single user-provided db_path and pre-open a session.
        This reduces LLM mistakes and ensures subsequent tool calls reuse session_id.
        """
        p = str(db_path or "").strip()
        if not p:
            raise ValueError("db_path must be a non-empty string")
        # Normalize to an absolute path rooted at workspace_dir when relative.
        if os.path.isabs(p):
            self._allowed_db_path = os.path.abspath(p)
        else:
            self._allowed_db_path = os.path.abspath(str(self.workspace_dir / p))
        self.ensure_session(read_only=bool(read_only))

    def ensure_session(self, *, read_only: bool = True) -> str:
        """
        Ensure a session is opened for the allowed db_path and cache session_id.
        """
        if self._default_session_id:
            return self._default_session_id
        if not self._allowed_db_path:
            raise ValueError("No db_path configured. Call set_db_context(db_path=...) first.")

        res = self._mcp_client.call_tool(
            "open_session",
            {
                "db_path": self._allowed_db_path,
                "db_name": None,
                "db_root": None,
                "db_files_dir": None,
                "db_edit_dir": None,
                "read_only": bool(read_only),
            },
        )
        # MCP server returns JSON text in content[0].text (or sets structuredContent).
        payload: Any = None
        try:
            txt = ""
            if res.structuredContent is not None:
                payload = res.structuredContent
            else:
                for c in (res.content or []):
                    if getattr(c, "type", None) == "text":
                        txt = str(getattr(c, "text", "") or "")
                        break
                if getattr(res, "isError", False):
                    # Server signaled an error; surface raw text (often not JSON).
                    raise RuntimeError(txt.strip() or "open_session failed (no details)")
                payload = json.loads(txt) if txt else {}
        except RuntimeError:
            raise
        except Exception:
            payload = {}

        sid = str(payload.get("session_id") or "").strip() if isinstance(payload, dict) else ""
        if not sid:
            # Include the raw MCP result shape if parsing fails.
            try:
                is_err = bool(getattr(res, "isError", False))
                txts = []
                for c in (res.content or []):
                    if getattr(c, "type", None) == "text":
                        t = str(getattr(c, "text", "") or "").strip()
                        if t:
                            txts.append(t)
                preview = " | ".join(txts[:2])
                raise RuntimeError(f"open_session failed (isError={is_err}): {preview or payload}")
            except RuntimeError:
                raise
            except Exception:
                raise RuntimeError(f"open_session failed: {payload}")
        self._default_session_id = sid
        return sid

    def _invalidate_session(self) -> None:
        """Clear the cached session_id so the next open_session call actually re-opens."""
        self._default_session_id = None

    def _build_llm(self, usage_id: str = "agent", completions_dir: Path | None = None) -> LLM:
        litellm_extra_body: dict[str, object] = {}
        model_lower = str(self.model_name or "").lower()
        if "deepseek" in model_lower:
            litellm_extra_body["max_tokens"] = self.max_output_tokens

        log_dir = completions_dir or self.completions_dir

        llm_kwargs: dict[str, object] = dict(
            usage_id=usage_id,
            model=self.model_name,
            stream=False,
            temperature=self.temperature,
            top_p=1,
            max_output_tokens=self.max_output_tokens,
            litellm_extra_body=litellm_extra_body,
            log_completions=True,
            log_completions_folder=str(log_dir),
        )
        apply_litellm_provider_kwargs(
            llm_kwargs,
            model=str(self.model_name or ""),
            api_key=self.api_key,
            base_url=self.base_url,
            enable_portkey_openai_cache=True,
        )

        return LLM(**llm_kwargs)

    def _build_mcp_tool_definitions(self) -> None:
        """Build ToolDefinition instances for every MCP tool and store them locally.

        This replaces the old global-registry + factory approach.  Each
        ``DbAgentRuntime`` instance builds its own ``ToolDefinition`` objects
        with ``self._mcp_client`` baked directly into the executor.  The
        definitions are later injected into ``agent._tools`` in
        ``_pre_init_agent()`` — completely bypassing the global registry and
        its internal ``ThreadPoolExecutor``, which was the source of all
        thread-local / cross-rollout conflicts.
        """
        tools = self._mcp_client.list_tools()
        self._mcp_tools = [t.name for t in tools if isinstance(t.name, str) and t.name.strip()]
        self._pre_built_tool_defs: list = []

        for t in tools:
            tool_name = str(t.name or "").strip()
            if not tool_name:
                continue
            if tool_name in self.blocked_tools:
                continue

            action_model = build_action_model(t, namespace=self._model_namespace)
            desc = str(t.description or "").strip()
            required_keys = []
            try:
                schema = t.inputSchema or {}
                if isinstance(schema, dict) and isinstance(schema.get("required"), list):
                    required_keys = [str(x) for x in schema["required"] if isinstance(x, (str, int, float, bool))]
            except Exception:
                required_keys = []

            executor = McpToolExecutor(
                self._mcp_client,           # captured directly — no thread-local needed
                tool_name,
                required_keys=required_keys,
                session_id_getter=lambda: self._default_session_id,
                session_id_invalidator=self._invalidate_session,
                allowed_db_path_getter=lambda: self._allowed_db_path,
            )
            tool_def = type(
                f"Mcp_{tool_name}_Tool",
                (McpProxyTool,),
                {"name": tool_name},
            )(
                description=desc,
                action_type=action_model,
                observation_type=McpToolObservation,
                executor=executor,
            )
            self._pre_built_tool_defs.append(tool_def)

    def _pre_init_agent(self, state: Any) -> None:
        """Inject pre-built tools into the agent, bypassing _initialize.

        ``agent._initialize()`` resolves ``Tool`` specs via the global registry
        inside a ``ThreadPoolExecutor`` — making thread-local storage unusable.
        By pre-populating ``agent._tools`` and setting ``agent._initialized``
        here (called from within ``run()`` in the correct execution context),
        we skip the registry entirely.  Default tools (FinishTool / ThinkTool)
        still require the conversation state and are created here too.
        """
        from openhands.sdk.tool import BUILT_IN_TOOL_CLASSES

        all_tools = list(self._pre_built_tool_defs)
        for tool_name in ["FinishTool", "ThinkTool"]:
            tool_class = BUILT_IN_TOOL_CLASSES.get(tool_name)
            if tool_class is not None:
                try:
                    all_tools.extend(tool_class.create(state))
                except Exception:
                    pass

        self.agent._tools = {td.name: td for td in all_tools}
        self.agent._initialized = True

    def _build_main_agent(self) -> Agent:
        pruning_condenser = ObservationPruningCondenser(
            prune_threshold=2_000,   # prune consumed obs larger than 2K chars
            keep_recent_obs=2,       # always keep the 2 most-recent data results intact
        )
        summarising_condenser = LLMSummarizingCondenser(
            llm=self.main_llm.model_copy(update={"usage_id": "condenser"}),
            max_size=80,
            keep_first=2,
        )
        condenser = PipelineCondenser(condensers=[pruning_condenser, summarising_condenser])

        # Use DB agent prompt as the full system prompt (not OpenHands default + suffix).
        ctx = AgentContext(system_message_suffix="")
        if isinstance(self.system_prompt_path, str) and self.system_prompt_path.strip():
            prompt_path = str(Path(self.system_prompt_path.strip()).resolve())
        else:
            prompt_path = str((Path(__file__).resolve().parent / "prompts" / "agentic_db.md").resolve())
        # tools=[] and include_default_tools=[] because we inject all ToolDefinition
        # instances directly via _pre_init_agent() before send_message() is called.
        # This bypasses the global registry and its ThreadPoolExecutor completely.
        return Agent(
            llm=self.main_llm,
            tools=[],
            include_default_tools=[],
            condenser=condenser,
            agent_context=ctx,
            system_prompt_filename=prompt_path,
            system_prompt_kwargs={"cli_mode": True, "auto_mode": bool(self.auto_mode)},
        )

    def create_conversation(self) -> LocalConversation:
        """Create the main agent's conversation for long-running sessions."""
        persistence_dir = self.run_dir / "logs" / "conversations" / "main_agent"
        persistence_dir.mkdir(parents=True, exist_ok=True)

        callbacks = []
        if self._on_action is not None:
            _cb = self._on_action
            _count = [0]

            def _action_callback(event: Any) -> None:
                try:
                    from openhands.sdk.event.llm_convertible import ActionEvent
                    if isinstance(event, ActionEvent):
                        _count[0] += 1
                        _cb(event.tool_name, _count[0])
                except Exception:
                    pass

            callbacks = [_action_callback]

        self.conversation = LocalConversation(
            agent=self.agent,
            workspace=str(self.workspace_dir.resolve()),
            persistence_dir=str(persistence_dir),
            max_iteration_per_run=self.max_iteration_per_run,
            delete_on_close=False,
            callbacks=callbacks if callbacks else None,
        )
        # Keep a single confirmation layer.
        # - `llm_map` / `llm_reduce` already have MCP-level plan -> user confirm -> run.
        # - `edit_duckdb` has no plan_id safety, so keep OpenHands confirmation for it.
        self.conversation.state.security_analyzer = _ToolNameSecurityAnalyzer(
            high_risk_tools={"edit_duckdb"}
        )
        self.conversation.state.confirmation_policy = ConfirmRisky(
            threshold=SecurityRisk.HIGH,
            confirm_unknown=True,
        )
        return self.conversation

    def run(self, prompt: str) -> str:
        """Send prompt and run the codebase agent.

        If OpenHands enters WAITING_FOR_CONFIRMATION, this method will prompt on
        stdin (when interactive) and only proceed if the user approves.
        """
        if self.conversation is None:
            self.create_conversation()

        # Inject pre-built ToolDefinitions into the agent before send_message
        # triggers _initialize.  This sets agent._initialized = True so the
        # SDK's ThreadPoolExecutor-based tool resolution is never invoked,
        # making concurrent rollouts safe without any thread-local storage.
        # Guard matches the SDK's own check so this is a no-op on subsequent turns.
        if not self.agent._initialized:
            self._pre_init_agent(self.conversation.state)

        self.conversation.send_message(prompt, sender="user")
        while True:
            self.conversation.run()
            if (
                self.conversation.state.execution_status
                == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
            ):
                if self.auto_mode:
                    # Batch mode: implicitly approve and keep running.
                    continue
                try:
                    import sys

                    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
                except Exception:
                    interactive = False

                if not interactive:
                    return (
                        "Agent is waiting for confirmation to run a risky tool call, "
                        "but stdin is non-interactive. Re-run in an interactive terminal "
                        "to approve or deny."
                    )

                ans = input("Approve pending risky actions? [y/N] ").strip().lower()
                if ans in {"y", "yes"}:
                    continue  # implicit confirmation: next run() executes pending actions
                return "Cancelled (risky actions not approved)."
            break
        return get_agent_final_response(self.conversation.state.events)

    def get_llm_metrics(self) -> dict:
        """Aggregate cost and token usage from all LLM instances (agent + MCP tools)."""
        total_cost = 0.0
        total_prompt = 0
        total_completion = 0
        for llm in self._llms:
            total_cost += llm.metrics.accumulated_cost
            if llm.metrics.accumulated_token_usage:
                total_prompt += llm.metrics.accumulated_token_usage.prompt_tokens
                total_completion += llm.metrics.accumulated_token_usage.completion_tokens
        mcp = self._get_mcp_llm_token_usage()
        mcp_prompt = int(mcp.get("prompt_tokens") or 0)
        mcp_completion = int(mcp.get("completion_tokens") or 0)
        return {
            "total_cost": total_cost,
            # Backward-compatible keys: these are now TOTAL (agent + MCP).
            "prompt_tokens": int(total_prompt + mcp_prompt),
            "completion_tokens": int(total_completion + mcp_completion),
            # Breakdown (agent-only vs MCP-only).
            "agent_prompt_tokens": int(total_prompt),
            "agent_completion_tokens": int(total_completion),
            "mcp_prompt_tokens": int(mcp_prompt),
            "mcp_completion_tokens": int(mcp_completion),
            "mcp_total_tokens": int(mcp.get("total_tokens") or (mcp_prompt + mcp_completion)),
        }

    def _get_mcp_llm_token_usage(self) -> dict[str, int]:
        """
        Sum token usage reported by MCP tool calls within this run.

        The MCP server writes JSONL logs under: <run_dir>/logs/mcp_server/
        Each successful tool call may include a top-level `llm_token_usage` object:
          {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
        """
        logs_dir = (self.run_dir / "logs" / "mcp_server").resolve()
        if not logs_dir.exists() or not logs_dir.is_dir():
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        try:
            for p in sorted(logs_dir.glob("*.jsonl")):
                try:
                    with p.open("r", encoding="utf-8") as f:
                        for line in f:
                            s = (line or "").strip()
                            if not s:
                                continue
                            try:
                                ev = json.loads(s)
                            except Exception:
                                continue
                            if not isinstance(ev, dict):
                                continue
                            tu = ev.get("llm_token_usage")
                            if not isinstance(tu, dict) or not tu:
                                continue
                            try:
                                it = tu.get("input_tokens")
                                ot = tu.get("output_tokens")
                                tt = tu.get("total_tokens")
                                if it is not None:
                                    input_tokens += int(it)
                                if ot is not None:
                                    output_tokens += int(ot)
                                if tt is not None:
                                    total_tokens += int(tt)
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # If total_tokens wasn't logged consistently, compute a best-effort total.
        if total_tokens <= 0:
            total_tokens = int(input_tokens + output_tokens)

        return {
            "prompt_tokens": int(input_tokens),
            "completion_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    def close(self) -> None:
        """Cleanup resources."""
        if self.conversation:
            try:
                self.conversation.close()
            except Exception:
                pass
        try:
            self._mcp_client.close()
        except Exception:
            pass


# Backward-compatible alias: old pipeline imports `CodebaseRuntime`.
CodebaseRuntime = DbAgentRuntime
