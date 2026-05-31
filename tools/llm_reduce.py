from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from tools.run_sql import exec_readonly_sql
from tools.tool_base import ToolPlugin, ToolRuntime
from utils.db import attach_extra_required_dbs, is_readonly_sql, run_readonly_query_on_conn
from utils.llm import (
    get_llmop_model,
    heuristic_tokens_from_chars,
    llmop_call,
    llmop_postprocess_output_text,
    payload_for_error,
    plan_fingerprint_payload,
    plan_id,
    truncate_cell,
)


_NON_SEMANTIC_PLAN_KEYS = {
    # These options affect execution performance/UX, not the semantic question being confirmed.
    "preview_sql",
    "limit_rows",
}



def _fetch_context(
    *,
    rt: ToolRuntime,
    context_sql: str,
    db_path: Optional[str],
    limit_rows: int,
) -> Dict[str, Any]:
    """
    Execute a read-only SELECT/WITH and return {columns, rows, row_count, limit_rows}.
    """
    if getattr(rt, "session_manager", None) is not None and getattr(rt, "session_id", None):
        sess = rt.session_manager.get(rt.session_id)  # type: ignore[union-attr]
        with sess.lock:
            attach_extra_required_dbs(
                conn=sess.conn,
                required_dbs=list(rt.required_dbs),
                db_files_dir=str(rt.db_files_dir or ""),
            )
            out = run_readonly_query_on_conn(conn=sess.conn, sql_text=context_sql, limit_rows=int(limit_rows))
            return out
    return exec_readonly_sql(
        sql_text=context_sql,
        required_dbs=list(rt.required_dbs),
        db_files_dir=str(rt.db_files_dir or ""),
        db_edit_dir=str(rt.db_edit_dir or ""),
        db_path=db_path,
        limit_rows=int(limit_rows),
    )


def _prompt_for_reduce(
    *,
    question: str,
    context: Dict[str, Any],
    output_type: str,
    options: Optional[str],
) -> str:
    cols = context.get("columns") or []
    rows = context.get("rows") or []
    payload = {"columns": cols, "rows": rows}
    ctx_json = json.dumps(payload, ensure_ascii=False)

    constraints = "Return ONLY the answer as plain text."
    if output_type == "boolean":
        constraints = "Return ONLY TRUE or FALSE."
    elif output_type == "number":
        constraints = "Return ONLY a number (no extra text). If unknown, return NULL."
    elif output_type == "json":
        constraints = "Return ONLY a JSON value (object/array/string/number/boolean/null)."

    opts = ""
    if isinstance(options, str) and options.strip():
        opts = f"\nAllowed options (choose one if applicable): {options.strip()}\n"

    return (
        "You are answering a question using database query results as context.\n"
        f"{constraints}\n"
        "If the answer cannot be determined from the context, return NULL.\n"
        f"{opts}"
        "\n"
        f"Question:\n{question.strip()}\n"
        "\n"
        f"Context (JSON with columns + rows):\n{ctx_json}\n"
    )


def _coerce_answer(answer_text: str, *, output_type: str) -> Any:
    t = llmop_postprocess_output_text(answer_text or "")
    if not t:
        return None
    if t.strip().lower() == "null":
        return None

    if output_type == "text":
        return t
    if output_type == "boolean":
        s = t.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
        return None
    if output_type == "number":
        try:
            if re.fullmatch(r"[+-]?\d+", t.strip()):
                return int(t.strip())
            return float(t.strip())
        except Exception:
            return None
    if output_type == "json":
        try:
            return json.loads(t)
        except Exception:
            return None
    return t


def _llm_reduce_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "llm_reduce",
        "description": (
            "Fetch rows with a SQL subquery, then have an LLM answer a question over them — in one call. "
            "Use instead of iterating SQL when the answer requires world knowledge, semantic reasoning, or computation SQL cannot express. "
            "Filter aggressively in context_sql first; call once after schema exploration rather than retrying SQL. "
            "action='run' (default) executes immediately; action='plan' previews token cost only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": ["string", "null"],
                    "enum": ["plan", "run", None],
                    "description": "Default 'run'. Use 'plan' to preview token cost before executing.",
                },
                "question": {"type": "string", "description": "The question to answer over the context rows."},
                "context_sql": {
                    "type": "string",
                    "description": "Read-only SELECT/WITH that produces context rows. Filter aggressively with SQL before passing rows to the LLM.",
                },
                "output_type": {
                    "type": ["string", "null"],
                    "enum": ["text", "number", "boolean", "json", None],
                    "description": "Expected output type. Default: text.",
                },
                "options": {
                    "type": ["string", "null"],
                    "description": "Optional allowed options / rubric for the answer (free-form string).",
                },
                "limit_rows": {
                    "type": ["integer", "null"],
                    "description": "Max context rows to include (1-500). If null, defaults to 50.",
                },
                "max_cell_chars": {
                    "type": ["integer", "null"],
                    "description": "Max characters per cell in context passed to LLM (10-500). If null, defaults to 120.",
                },
                "confirm": {
                    "type": ["boolean", "null"],
                    "description": "Only required when using the two-step plan→run flow: must be true.",
                },
                "plan_id": {
                    "type": ["string", "null"],
                    "description": "Only required when using the two-step plan→run flow: must match the plan_id from action='plan'.",
                },
                "db_path": {
                    "type": ["string", "null"],
                    "description": "Optional DB path for fetching (must be under db_files_dir or db_edit_dir).",
                },
            },
            "required": ["question", "context_sql"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _llm_reduce_handler_factory(rt: ToolRuntime) -> Callable[[Dict[str, Any]], Any]:
    def handler(args: Dict[str, Any]) -> Any:
        action = str(args.get("action") or "run").strip().lower()
        if action not in ("plan", "run"):
            raise ValueError("action must be 'plan' or 'run'")

        question = str(args.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")
        if len(question) > 4000:
            raise ValueError("question is too long (max 4000 chars)")

        context_sql_raw = str(args.get("context_sql") or "").strip()
        if not context_sql_raw:
            raise ValueError("context_sql is required")
        if not is_readonly_sql(context_sql_raw):
            raise ValueError("Only read-only SELECT/WITH SQL is allowed for context_sql.")
        context_sql = context_sql_raw.rstrip().rstrip(";").strip()

        output_type_raw = args.get("output_type")
        output_type = "text" if output_type_raw is None else str(output_type_raw).strip().lower()
        if output_type not in ("text", "number", "boolean", "json"):
            raise ValueError("output_type must be one of: text, number, boolean, json (or null)")

        options = args.get("options")
        options_s = str(options) if isinstance(options, str) else None
        if options_s is not None and len(options_s) > 4000:
            raise ValueError("options is too long (max 4000 chars)")

        limit_rows_raw = args.get("limit_rows")
        limit_rows = int(limit_rows_raw) if limit_rows_raw is not None else 50
        limit_rows = max(1, min(limit_rows, 500))

        max_cell_chars_raw = args.get("max_cell_chars")
        max_cell_chars = int(max_cell_chars_raw) if max_cell_chars_raw is not None else 120
        max_cell_chars = max(10, min(max_cell_chars, 500))

        db_path = args.get("db_path")
        db_path_norm = str(db_path).strip() if isinstance(db_path, str) and db_path.strip() else None

        model_name = get_llmop_model()

        # Fetch context and truncate for prompt.
        ctx = _fetch_context(rt=rt, context_sql=context_sql, db_path=db_path_norm, limit_rows=limit_rows)
        cols = ctx.get("columns") or []
        rows = ctx.get("rows") or []

        truncated_rows: List[List[Any]] = []
        for r in rows:
            if not isinstance(r, list):
                continue
            truncated_rows.append([truncate_cell(v, max_chars=max_cell_chars) for v in r])

        ctx_for_prompt = {
            "columns": cols,
            "rows": truncated_rows,
            "row_count": int(ctx.get("row_count") or len(truncated_rows)),
            "limit_rows": int(ctx.get("limit_rows") or limit_rows),
        }

        prompt = _prompt_for_reduce(
            question=question,
            context=ctx_for_prompt,
            output_type=output_type,
            options=options_s,
        )
        prompt_chars = len(prompt)
        est_tokens = heuristic_tokens_from_chars(prompt_chars)

        plan_payload = {
            "model": model_name,
            "question": question,
            "context_sql": context_sql,
            "output_type": output_type,
            "options": options_s or "",
            "limit_rows": int(limit_rows),
            "max_cell_chars": int(max_cell_chars),
            "columns": cols,
        }
        fp_payload = plan_fingerprint_payload(plan_payload, non_semantic_keys=_NON_SEMANTIC_PLAN_KEYS)
        pid = plan_id(fp_payload)

        if action == "plan":
            return {
                "action": "plan",
                "plan_id": pid,
                "model": model_name,
                "output_type": output_type,
                "limit_rows": int(limit_rows),
                "max_cell_chars": int(max_cell_chars),
                "plan_fingerprint_payload": fp_payload,
                "context_row_count": int(len(truncated_rows)),
                "context_columns": cols,
                "estimated_prompt_chars": int(prompt_chars),
                "estimated_tokens": int(est_tokens),
                "note": "To execute, call again with action='run', confirm=true, and the same plan_id.",
            }

        # action == "run"
        plan_id_in = args.get("plan_id")
        if isinstance(plan_id_in, str) and plan_id_in.strip():
            # Two-step flow: validate confirm + plan_id fingerprint.
            confirm = args.get("confirm")
            if confirm is not True:
                raise ValueError("When plan_id is provided, confirm=true is required.")
            if plan_id_in.strip() != pid:
                raise ValueError(
                    "plan_id mismatch.\n"
                    f"- provided: {plan_id_in.strip()}\n"
                    f"- expected: {pid}\n"
                    f"- current_plan_fingerprint_payload: {payload_for_error(fp_payload)}\n"
                    "Re-run action='plan' and use the returned plan_id."
                )

        out_text_raw, usage = llmop_call(model=model_name, prompt=prompt)
        out_text = llmop_postprocess_output_text(out_text_raw or "")
        typed = _coerce_answer(out_text, output_type=output_type)

        return {
            "action": "run",
            "plan_id": pid,
            "model": model_name,
            "output_type": output_type,
            "answer_text": out_text,
            "answer": typed,
            "token_usage": usage,
            "context": {
                "columns": cols,
                "rows": truncated_rows,
                "row_count": int(len(truncated_rows)),
                "limit_rows": int(limit_rows),
            },
            "prompt_stats": {
                "prompt_chars": int(prompt_chars),
                "estimated_tokens": int(est_tokens),
            },
        }

    return handler


TOOL_PLUGIN = ToolPlugin(tool=_llm_reduce_tool(), handler_factory=_llm_reduce_handler_factory)

