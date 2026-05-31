import json
from typing import Any, Callable, Dict, List, MutableSequence

from tools.edit_duckdb import TOOL_PLUGIN as _EDIT_DUCKDB_PLUGIN
from tools.column_mapping import TOOL_PLUGIN as _COLUMN_MAPPING_PLUGIN
from tools.describe_relation import TOOL_PLUGIN as _DESCRIBE_RELATION_PLUGIN
from tools.explain_sql import TOOL_PLUGIN as _EXPLAIN_SQL_PLUGIN
from tools.list_relations import TOOL_PLUGIN as _LIST_RELATIONS_PLUGIN
from tools.llm_map import TOOL_PLUGIN as _LLM_MAP_PLUGIN
from tools.llm_reduce import TOOL_PLUGIN as _LLM_REDUCE_PLUGIN
from tools.materialize_temp import TOOL_PLUGIN as _MATERIALIZE_TEMP_PLUGIN
from tools.preview_relation import TOOL_PLUGIN as _PREVIEW_RELATION_PLUGIN
from tools.profile_query import TOOL_PLUGIN as _PROFILE_QUERY_PLUGIN
from tools.row_transform import TOOL_PLUGIN as _ROW_TRANSFORM_PLUGIN
from tools.run_sql import TOOL_PLUGIN as _RUN_SQL_PLUGIN
from tools.tool_base import ToolPlugin, ToolRuntime


# Tool-related prompts
CHAT_SYSTEM_PROMPT = (
    "You are a SQL annotator and assistant. Use the provided SQL and metadata context. "
    "When answering, be precise and cite table/column names. If something is unknown or not in context, say so."
    "Keep your answers short and concise. Use the same language as the user's question. "
)

CHAT_SYSTEM_PROMPT_STREAM = (
    "You are a SQL annotator and assistant. Use the provided SQL and metadata context. "
    "When answering, be precise and cite table/column names. If something is unknown or not in context, say so."
)

CHAT_RUN_SQL_PROMPT = (
    "Execute the current SQL against its DuckDB database(s) and show a small preview of results (first ~20 rows). "
    "Use the run_sql tool for read-only validation. If you need to edit/update the database, use edit_duckdb."
)


# Register tool plugins here. Add new tools by appending a ToolPlugin to this list.
TOOL_PLUGINS: List[ToolPlugin] = [
    # Read-only SQL + inspection (orthogonal building blocks for stepwise agents)
    _RUN_SQL_PLUGIN,
    _EXPLAIN_SQL_PLUGIN,
    _LIST_RELATIONS_PLUGIN,
    _DESCRIBE_RELATION_PLUGIN,
    _PREVIEW_RELATION_PLUGIN,
    _MATERIALIZE_TEMP_PLUGIN,
    _PROFILE_QUERY_PLUGIN,
    # Persistent edits / write-back
    _EDIT_DUCKDB_PLUGIN,
    _ROW_TRANSFORM_PLUGIN,
    _COLUMN_MAPPING_PLUGIN,
    # LLM operators (reasoning and normalization)
    _LLM_MAP_PLUGIN,
    _LLM_REDUCE_PLUGIN,
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Returns all tool schemas to pass to the model (stable viewer interface).
    """
    return [p.tool for p in TOOL_PLUGINS]


def get_tool_handlers(runtime: ToolRuntime) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """
    Returns all tool handlers to execute function calls (stable viewer interface).
    """
    handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    for p in TOOL_PLUGINS:
        name = str(p.tool.get("name") or "")
        if not name:
            continue
        handlers[name] = p.handler_factory(runtime)
    return handlers


def js_string_literal(s: str) -> str:
    """
    Returns a valid JS string literal (including quotes) using JSON encoding.
    """
    return json.dumps(s, ensure_ascii=False)


def dispatch_function_tools(
    *,
    out_items: List[Any],
    input_list: MutableSequence[Dict[str, Any]],
    handlers: Dict[str, Callable[[Dict[str, Any]], Any]],
) -> bool:
    """
    Dispatches OpenAI Responses API function_call items to registered handlers and appends
    function_call_output items back onto input_list.

    Returns True if at least one tool call was handled.
    """
    handled_any = False
    for call in out_items:
        if getattr(call, "type", None) != "function_call":
            continue
        name = getattr(call, "name", "") or ""
        handler = handlers.get(name)
        if handler is None:
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", None),
                    "output": json.dumps({"error": f"Unknown tool: {name!r}"}),
                }
            )
            handled_any = True
            continue

        args_raw: Any = getattr(call, "arguments", "") or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}

        try:
            payload = handler(args)
        except Exception as e:
            payload = {"error": str(e)}

        try:
            output = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        except Exception:
            output = json.dumps({"error": "tool output serialization failed"}, ensure_ascii=False)

        input_list.append(
            {
                "type": "function_call_output",
                "call_id": getattr(call, "call_id", None),
                "output": output,
            }
        )
        handled_any = True

    return handled_any

