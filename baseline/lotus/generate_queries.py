#!/usr/bin/env python3
"""Stage 1: generate and persist fixed LOTUS programs for the SWAN benchmark."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    PROMPTS_DIR,
    REPO_ROOT,
    THIS_DIR,
    build_schema_prompt,
    load_db_tables,
    load_jsonl,
    resolve_duckdb_path,
    resolve_under_repo,
    select_records,
    sha256_file,
    sha256_text,
    write_jsonl_atomic,
)

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass


class InfrastructureError(RuntimeError):
    """Planner model/API failure after infrastructure retries are exhausted."""


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


SEMANTIC_VALUE_INSTRUCTION = load_prompt("semantic_value_instruction.md")
PLANNER_SYSTEM = load_prompt("planner_system.md").replace(
    "<SEMANTIC_VALUE_INSTRUCTION>", SEMANTIC_VALUE_INSTRUCTION
)
PLANNER_USER_TEMPLATE = load_prompt("planner_user.md")
ORACLE_BLUEPRINT_TEMPLATE = load_prompt("oracle_blueprint.md")
REPAIR_USER_TEMPLATE = load_prompt("repair_user.md")
EFFICIENCY_GUIDANCE = load_prompt("efficiency_guidance.md")
CODE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def is_bedrock(model: str) -> bool:
    return str(model or "").lower().startswith("bedrock/")


def planner_model() -> str:
    model = (os.getenv("PLANNER_MODEL") or os.getenv("AGENT_MODEL") or "").strip()
    if not model:
        raise RuntimeError("Planner model not configured (set PLANNER_MODEL or AGENT_MODEL).")
    return model


def planner_kwargs(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model}
    if not is_bedrock(model):
        api_key = (os.getenv("PLANNER_API_KEY") or os.getenv("AGENT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Planner API key required (set PLANNER_API_KEY or AGENT_API_KEY).")
        kwargs["api_key"] = api_key
        base_url = (os.getenv("PLANNER_BASE_URL") or os.getenv("AGENT_BASE_URL") or "").strip()
        if base_url:
            kwargs["base_url"] = base_url
    return kwargs


def stub_planner() -> str:
    """Deterministic offline program used to test generation plumbing."""
    return (
        "```python\n"
        "first = next(iter(tables.values()))\n"
        "result_df = first.head(5).copy()\n"
        "```\n"
    )


def call_planner(messages: list[dict[str, str]], *, dry_run: bool) -> str:
    if dry_run:
        return stub_planner()

    import litellm

    model = planner_model()
    reasoning_effort = (
        os.getenv("E1_PLANNER_REASONING_EFFORT") or "high"
    ).strip() or "high"
    try:
        response = litellm.completion(
            messages=messages,
            reasoning_effort=reasoning_effort,
            allowed_openai_params=["reasoning_effort"],
            **planner_kwargs(model),
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise InfrastructureError(f"planner model/API call failed: {exc}") from exc


def extract_code(text: str) -> str:
    match = CODE_PATTERN.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def oracle_blueprint(record: dict[str, Any]) -> str:
    sql = str(record.get("sql") or "").strip()
    blendsql = str(record.get("blendsql") or "").strip()
    if not sql and not blendsql:
        return ""
    return "\n" + ORACLE_BLUEPRINT_TEMPLATE.format(sql=sql, blendsql=blendsql) + "\n"


def generate_one(
    record: dict[str, Any],
    *,
    db_dir: Path,
    row_sample: int,
    max_attempts: int,
    oracle: bool,
    efficiency_guidance: bool,
    dry_run: bool,
) -> dict[str, Any]:
    question_id = str(record.get("question_id") or "").strip()
    db = str(record.get("db") or "").strip()
    query = str(record.get("query") or "")
    hint = str(record.get("hint") or "")
    required_columns = list(record.get("required_columns") or [])
    started = time.perf_counter()
    attempts = 0
    program = ""
    planner_messages_sha256 = ""
    generation_error: str | None = None

    try:
        tables = load_db_tables(resolve_duckdb_path(db, db_dir))
        schema = build_schema_prompt(db, tables, row_sample)
        required_block = ""
        if required_columns:
            required_block = (
                "\nREQUIRED OUTPUT COLUMNS (result_df must have exactly these, "
                f"named and ordered): {required_columns}\n"
            )
        system_prompt = PLANNER_SYSTEM
        if efficiency_guidance:
            system_prompt += "\n\n" + EFFICIENCY_GUIDANCE
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": PLANNER_USER_TEMPLATE.format(
                    query=query,
                    hint=hint,
                    required=required_block,
                    schema=schema,
                    blueprint=oracle_blueprint(record) if oracle else "",
                ),
            },
        ]
        planner_messages_sha256 = sha256_text(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )

        for attempt in range(1, max(1, max_attempts) + 1):
            attempts = attempt
            raw = call_planner(messages, dry_run=dry_run)
            candidate = extract_code(raw)
            try:
                compile(candidate, f"<lotus-program:{question_id}>", "exec")
            except SyntaxError:
                generation_error = "syntax_error"
                if attempt >= max(1, max_attempts):
                    program = candidate
                    break
                syntax_traceback = traceback.format_exc(limit=4)
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": REPAIR_USER_TEMPLATE.format(
                                traceback=syntax_traceback[-2000:]
                            ),
                        },
                    ]
                )
                continue
            program = candidate
            generation_error = None
            break
    except InfrastructureError as exc:
        generation_error = f"infrastructure: {exc}"
    except Exception as exc:
        generation_error = f"fatal: {type(exc).__name__}: {exc}"

    elapsed = round(time.perf_counter() - started, 3)
    result = {
        "question_id": question_id,
        "db": db,
        "query": query,
        "hint": hint,
        "required_columns": required_columns,
        "oracle": bool(oracle),
        "efficiency_guidance": bool(efficiency_guidance),
        "program": program,
        "generation_error": generation_error,
        "generation_attempts": attempts,
        "generation_wall_time_s": elapsed,
        "generation_timing_accounting": "excluded_from_execution_metrics",
        "program_sha256": sha256_text(program),
        "planner_messages_sha256": planner_messages_sha256,
    }
    print(
        f"[{question_id}] generated attempts={attempts} error={generation_error} "
        f"t={elapsed}s",
        flush=True,
    )
    return result


def generate_with_query_retries(
    record: dict[str, Any],
    *,
    max_query_retries: int,
    query_retry_delay_s: float,
    **generation_kwargs: Any,
) -> dict[str, Any]:
    """Retry a failed query generation from its original prompt."""
    started = time.perf_counter()
    errors: list[str] = []
    result: dict[str, Any] = {}
    attempts = max(1, int(max_query_retries) + 1)
    for query_attempt in range(1, attempts + 1):
        result = generate_one(record, **generation_kwargs)
        error = result.get("generation_error")
        if error is None:
            break
        errors.append(str(error))
        if query_attempt < attempts:
            print(
                f"[{result['question_id']}] query generation failed; retrying "
                f"{query_attempt}/{max_query_retries} after "
                f"{query_retry_delay_s:g}s",
                flush=True,
            )
            time.sleep(max(0.0, float(query_retry_delay_s)))
    result["query_generation_attempts"] = query_attempt
    result["query_generation_retries"] = query_attempt - 1
    result["query_generation_errors"] = errors
    result["generation_wall_time_s_total"] = round(
        time.perf_counter() - started, 3
    )
    return result


def parse_args() -> argparse.Namespace:
    query_default = os.getenv("SWAN_QUERY_FILE") or str(REPO_ROOT / "swan" / "evaluation.jsonl")
    db_default = os.getenv("DB_FILES_DIR") or str(REPO_ROOT / "swan" / "database")
    row_sample_default = int((os.getenv("E1_ROW_SAMPLE_SIZE") or "20").strip() or "20")
    attempts_default = int(
        (os.getenv("E1_MAX_TRANSLATION_ATTEMPTS") or "3").strip() or "3"
    )

    parser = argparse.ArgumentParser(description="Generate fixed LOTUS programs for SWAN.")
    parser.add_argument("--query_file", default=query_default, help="SWAN evaluation JSONL.")
    parser.add_argument("--db_dir", default=db_default, help="Directory containing SWAN DuckDB files.")
    parser.add_argument("--out_file", default="", help="Output JSONL; defaults by --oracle mode.")
    parser.add_argument("--oracle", action="store_true", help="Include gold SQL and BlendSQL in the planner prompt.")
    efficiency_group = parser.add_mutually_exclusive_group(required=True)
    efficiency_group.add_argument("--efficiency-guidance", dest="efficiency_guidance", action="store_true", help="Append fixed-plan efficiency guidance to the planner system prompt.")
    efficiency_group.add_argument("--no-efficiency-guidance", dest="efficiency_guidance", action="store_false", help="Generate from the base DSL/fairness prompt without efficiency guidance.")
    parser.add_argument("--row_sample", type=int, default=row_sample_default, help="Rows sampled per table.")
    parser.add_argument("--max_attempts", type=int, default=attempts_default, help="Planner attempts, with retries only for static syntax errors.")
    parser.add_argument("--max_query_retries", type=int, default=int((os.getenv("E1_MAX_QUERY_GENERATION_RETRIES") or "3").strip() or "3"), help="Fresh query-generation retries after final generation failure.")
    parser.add_argument("--query_retry_delay_s", type=float, default=float((os.getenv("E1_QUERY_GENERATION_RETRY_DELAY_S") or "5").strip() or "5"), help="Fixed delay between fresh query-generation retries.")
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N matched questions.")
    parser.add_argument("--run_id", action="append", default=[], help="Only these question IDs; repeatable.")
    parser.add_argument("--run_db", action="append", default=[], help="Only these database names; repeatable.")
    parser.add_argument("--resume", action="store_true", help="Keep successful records already present in the output JSONL and continue missing/failed questions.")
    parser.add_argument("--dry_run", action="store_true", help="Use a deterministic planner stub; never access the network.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "oracle" if args.oracle else "fair"
    guidance = "efficiency" if args.efficiency_guidance else "no_efficiency"
    default_name = f"generated_queries_{mode}_{guidance}.jsonl"
    out_file = resolve_under_repo(args.out_file) if args.out_file else (THIS_DIR / default_name)
    query_file = resolve_under_repo(args.query_file)
    records = select_records(
        load_jsonl(query_file),
        run_ids=args.run_id,
        run_dbs=args.run_db,
        limit=args.limit,
    )
    db_dir = resolve_under_repo(args.db_dir)
    print(
        f"LOTUS generation stage: {len(records)} queries | out_file={out_file} | "
        f"oracle={args.oracle} | efficiency_guidance={args.efficiency_guidance} | "
        f"dry_run={args.dry_run}",
        flush=True,
    )

    generated_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and out_file.is_file():
        for existing in load_jsonl(out_file):
            if (
                bool(existing.get("oracle")) == bool(args.oracle)
                and bool(existing.get("efficiency_guidance"))
                == bool(args.efficiency_guidance)
            ):
                generated_by_id[str(existing.get("question_id") or "")] = existing
    resumed = sum(
        1
        for record in records
        if (
            str(record.get("question_id") or "") in generated_by_id
            and generated_by_id[str(record.get("question_id") or "")].get(
                "generation_error"
            )
            is None
        )
    )
    if resumed:
        print(f"Resuming with {resumed} successful records already persisted.", flush=True)
    for record in records:
        question_id = str(record.get("question_id") or "")
        existing = generated_by_id.get(question_id)
        if existing is None or existing.get("generation_error") is not None:
            generated_by_id[question_id] = generate_with_query_retries(
                    record,
                    max_query_retries=max(0, args.max_query_retries),
                    query_retry_delay_s=max(0.0, args.query_retry_delay_s),
                    db_dir=db_dir,
                    row_sample=max(0, args.row_sample),
                    max_attempts=max(1, args.max_attempts),
                    oracle=bool(args.oracle),
                    efficiency_guidance=bool(args.efficiency_guidance),
                    dry_run=bool(args.dry_run),
                )
        generated = [
            generated_by_id[str(source.get("question_id") or "")]
            for source in records
            if str(source.get("question_id") or "") in generated_by_id
        ]
        write_jsonl_atomic(out_file, generated)

    generated = [
        generated_by_id[str(record.get("question_id") or "")]
        for record in records
        if str(record.get("question_id") or "") in generated_by_id
    ]

    manifest = {
        "program_file": str(out_file),
        "program_file_sha256": sha256_file(out_file),
        "query_file": str(query_file),
        "query_file_sha256": sha256_file(query_file),
        "db_dir": str(db_dir),
        "records": len(generated),
        "oracle": bool(args.oracle),
        "efficiency_guidance": bool(args.efficiency_guidance),
        "dry_run": bool(args.dry_run),
        "planner_model": "stub" if args.dry_run else planner_model(),
        "planner_reasoning_effort": (
            os.getenv("E1_PLANNER_REASONING_EFFORT") or "high"
        ).strip() or "high",
        "row_sample": int(args.row_sample),
        "row_sample_max_chars": int(
            (os.getenv("E1_ROW_SAMPLE_MAX_CHARS") or "6000").strip() or "6000"
        ),
        "max_generation_attempts": int(args.max_attempts),
        "max_query_generation_retries": int(args.max_query_retries),
        "query_generation_retry_delay_s": float(args.query_retry_delay_s),
        "prompt_file_sha256": {
            path.name: sha256_file(path)
            for path in sorted(PROMPTS_DIR.glob("*.md"))
        },
        "generation_wall_time_s_total": round(
            sum(float(record.get("generation_wall_time_s", 0) or 0) for record in generated),
            3,
        ),
        "generation_time_is_execution_metric": False,
    }
    out_file.with_suffix(out_file.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Fixed programs written to {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
