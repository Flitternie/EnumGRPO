from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import _check_no_external_access

logger = logging.getLogger(__name__)

# Lazy-loaded module-level cache for the BlendSQL ingredient model.
_CACHED_MODEL: Any = None
_CACHED_MODEL_KEY: Optional[str] = None


def _get_or_build_model() -> Any:
    """Return a cached BlendSQL ModelBase instance, building on first call."""
    global _CACHED_MODEL, _CACHED_MODEL_KEY

    model_name = (os.getenv("LLMOP_MODEL") or "").strip()
    cache_key = model_name

    if _CACHED_MODEL is not None and _CACHED_MODEL_KEY == cache_key:
        return _CACHED_MODEL

    from baseline.blendsql.run import build_model  # type: ignore

    model, _ = build_model(model_name=model_name, caching=True)
    _CACHED_MODEL = model
    _CACHED_MODEL_KEY = cache_key
    return model


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _json_safe(v: Any) -> Any:
    return None if _is_nan(v) else v


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


def _log_token_usage(
    *,
    blendsql: str,
    token_usage: Dict[str, Any],
    num_generation_calls: int,
) -> None:
    """Emit a structured log line for BlendSQL LLM-op token usage."""
    preview = blendsql[:120].replace("\n", " ")
    summary = json.dumps(
        {"token_usage": token_usage, "num_generation_calls": num_generation_calls},
        separators=(",", ":"),
    )
    logger.info("[run_blendsql] llm_op_token_usage query=%r %s", preview, summary)


def _run_blendsql_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "run_blendsql",
        "description": (
            "Execute a BlendSQL query against the database. "
            "Accepts pure SQL (no ingredients required) or SQL extended with LLM-powered ingredients for semantic reasoning, fuzzy matching, or external knowledge:\n"
            "Do NOT use glob() or ATTACH — querying files outside the provided database is not allowed.\n"
            "- LLMMap('question', 'alias.column') — maps an LLM predicate over every row; 'alias.column' must use the exact table alias from your FROM/JOIN clause. "
            "Optional: options=('val1','val2',...) to constrain output; return_type='int'|'float'|'bool' for numeric output.\n"
            "- LLMQA('question', (SELECT ...)) — returns a single scalar answer from the LLM using the subquery rows as context. "
            "Always provide a subquery to ground the answer in actual database data. "
            "Omit the subquery only when you have confirmed the required data is missing from the schema. "
            "Optional: options=('val1','val2',...) to constrain to a closed set; return_type='int'|'float'.\n"
            "- LLMJoin(left_on='t1.col', right_on='t2.col') — fuzzy-joins two tables via the LLM when key values do not share an exact format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "blendsql": {
                    "type": "string",
                    "description": (
                        "The BlendSQL query string. Can be pure SQL or SQL enhanced with "
                        "ingredient expressions: {{ LLMMap(...) }}, {{ LLMQA(...) }}, {{ LLMJoin(...) }}. "
                        "See the tool description for full ingredient signatures and constraints."
                    ),
                },
                "output_path": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional .csv path to write results to (relative to workspace root, or absolute within it). "
                        "Must end with .csv."
                    ),
                },
                "include_header": {
                    "type": ["boolean", "null"],
                    "description": "For CSV output: whether to include a header row. Default true.",
                },
                "timeout_s": {
                    "type": ["integer", "null"],
                    "description": "Execution timeout in seconds. Default 180. Set 0 to disable.",
                },
            },
            "required": ["blendsql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _run_blendsql_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        blendsql_str = str(args.get("blendsql") or "").strip()
        if not blendsql_str:
            raise ValueError("blendsql must be a non-empty string")

        _check_no_external_access(blendsql_str)

        output_path_raw = args.get("output_path")
        _op = str(output_path_raw).strip() if isinstance(output_path_raw, str) else ""
        output_path = _op if _op and _op.lower() != "null" else None
        include_header_raw = args.get("include_header")
        include_header = True if include_header_raw is None else bool(include_header_raw)
        timeout_s = 0  # Disable SIGALRM — incompatible with asyncio thread pool

        # Get db_path from the session.
        if getattr(rt, "session_manager", None) is None or not getattr(rt, "session_id", None):
            raise RuntimeError("run_blendsql requires an active session. Call open_session first.")
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
        db_path = str(getattr(sess, "db_path", "") or "")
        if not db_path:
            raise RuntimeError("Session has no db_path set.")

        from baseline.blendsql.run import execute_blendsql  # type: ignore

        model = _get_or_build_model()

        result = execute_blendsql(
            db_path=db_path,
            blendsql=blendsql_str,
            model=model,
            verbose=False,
            timeout_s=timeout_s,
            capture_logs=True,
            tee_logs_to_console=False,
            duckdb_readonly=True,
        )

        # Build response.
        out: Dict[str, Any] = {}
        if result.df is not None:
            try:
                cols = [str(c) for c in list(result.df.columns)]
            except Exception:
                cols = []
            try:
                rows = [[_json_safe(v) for v in r] for r in result.df.values.tolist()]
            except Exception:
                rows = []
            out["columns"] = cols
            out["rows"] = rows[:200]  # Limit preview
            out["row_count"] = len(rows)
        else:
            out["columns"] = []
            out["rows"] = []
            out["row_count"] = 0

        out["token_usage"] = dict(result.token_usage) if result.token_usage else {}
        out["num_generation_calls"] = int(result.num_generation_calls)
        out["num_cache_hits"] = int(getattr(result, "num_cache_hits", 0) or 0)

        _log_token_usage(
            blendsql=blendsql_str,
            token_usage=out["token_usage"],
            num_generation_calls=out["num_generation_calls"],
        )

        if result.exec_stderr and result.exec_stderr.strip():
            out["stderr"] = result.exec_stderr.strip()[:2000]

        # Write CSV if requested.
        if output_path:
            out_path = _resolve_output_path(output_path)
            if result.df is not None:
                try:
                    result.df.to_csv(str(out_path), index=False)
                except TypeError:
                    result.df.to_csv(str(out_path))
                out["saved"] = {"path": str(out_path), "format": "csv"}
            else:
                _write_rows_to_csv(
                    path=out_path,
                    columns=out.get("columns", []),
                    rows=out.get("rows", []),
                    include_header=bool(include_header),
                )
                out["saved"] = {"path": str(out_path), "format": "csv"}

        return out

    return handler


def make_run_blendsql_plugin() -> ToolPlugin:
    return ToolPlugin(
        tool=_run_blendsql_tool(),
        handler_factory=_run_blendsql_handler_factory,
    )


TOOL_PLUGIN = make_run_blendsql_plugin()
