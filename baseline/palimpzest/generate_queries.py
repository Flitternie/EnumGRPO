#!/usr/bin/env python3
"""Stage 1: translate SWAN questions into fixed Palimpzest programs."""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    PZInfrastructureError,
    PZTableLoader,
    REPO_ROOT,
    THIS_DIR,
    build_pipeline_from_code,
    env,
    env_int,
    load_jsonl,
    resolve_db_path,
    sha256_file,
    sha256_text,
    write_jsonl,
)

PROMPTS_DIR = THIS_DIR / "prompts"
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


PLANNER_SYSTEM = load_prompt("planner_system.md")
PLANNER_USER_TEMPLATE = load_prompt("planner_user.md")
ORACLE_BLUEPRINT_TEMPLATE = load_prompt("oracle_blueprint.md")
REPAIR_USER_TEMPLATE = load_prompt("repair_user.md")
EFFICIENCY_GUIDANCE = load_prompt("efficiency_guidance.md")


def extract_code(text: str) -> str:
    match = CODE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def oracle_blueprint(query: dict[str, Any]) -> str:
    sql = str(query.get("sql") or "").strip()
    blendsql = str(query.get("blendsql") or "").strip()
    if not sql and not blendsql:
        return ""
    return "\n" + ORACLE_BLUEPRINT_TEMPLATE.format(sql=sql, blendsql=blendsql) + "\n"


def planner_model_id() -> str:
    model = env("PZ_PLANNER_MODEL") or env("AGENT_MODEL")
    if not model:
        raise RuntimeError("Set PZ_PLANNER_MODEL (or AGENT_MODEL) for query generation")
    return model


def planner_kwargs(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "reasoning_effort": env("PZ_PLANNER_REASONING_EFFORT", "high"),
    }
    timeout = env_int("PZ_PLANNER_TIMEOUT_S", env_int("LLMOP_TIMEOUT_S", 600))
    if timeout > 0:
        kwargs["timeout"] = timeout
    if model.lower().startswith("bedrock/"):
        return kwargs
    kwargs["allowed_openai_params"] = ["reasoning_effort"]
    api_key = env("PZ_PLANNER_API_KEY") or env("AGENT_API_KEY")
    base_url = env("PZ_PLANNER_BASE_URL") or env("AGENT_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return kwargs


def call_planner(messages: list[dict[str, str]]) -> str:
    import litellm

    kwargs = planner_kwargs(planner_model_id(), messages)
    try:
        response = litellm.completion(**kwargs)
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise PZInfrastructureError(f"planner model/API call failed: {exc}") from exc


def mock_planner(messages: list[dict[str, str]]) -> str:
    """Offline planner stub that still consumes the fully rendered live-schema prompt."""
    text = messages[-1]["content"]
    match = re.search(r"AVAILABLE TABLES:\s*\[([^\]]*)\]", text)
    table_name = "table"
    if match:
        names = [
            item.strip().strip("'\"")
            for item in match.group(1).split(",")
            if item.strip()
        ]
        if names:
            table_name = names[0]
    return (
        "```python\n"
        "def build_pipeline(db):\n"
        f"    return db.table({table_name!r}).limit(5)\n"
        "```\n"
    )


def build_messages(
    query: dict[str, Any],
    loader: PZTableLoader,
    *,
    sample_rows: int,
    oracle: bool,
    efficiency_guidance: bool,
) -> list[dict[str, str]]:
    required_columns = list(query.get("required_columns") or [])
    required = (
        "REQUIRED OUTPUT COLUMNS (final dataset must be exactly these, in order): "
        f"{required_columns}\n"
        if required_columns
        else ""
    )
    user_message = PLANNER_USER_TEMPLATE.format(
        schema=loader.schema_text(sample_rows=sample_rows),
        tables=loader.table_names(),
        query=str(query.get("query") or ""),
        hint=str(query.get("hint") or ""),
        required=required,
        blueprint=oracle_blueprint(query) if oracle else "",
    )
    system_prompt = PLANNER_SYSTEM
    if efficiency_guidance:
        system_prompt += "\n\n" + EFFICIENCY_GUIDANCE
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def generate_one(
    query: dict[str, Any],
    *,
    oracle: bool,
    efficiency_guidance: bool,
    sample_rows: int,
    max_attempts: int,
    validation_row_cap: int,
    planner_fn=call_planner,
) -> dict[str, Any]:
    qid = str(query.get("question_id") or "").strip()
    db_name = str(query.get("db") or "").strip()
    started = time.monotonic()
    attempts = 0
    program = ""
    planner_messages_sha256 = ""
    error: str | None = None
    loader: PZTableLoader | None = None
    try:
        loader = PZTableLoader(
            resolve_db_path(db_name), row_cap=max(1, validation_row_cap)
        )
        messages = build_messages(
            query,
            loader,
            sample_rows=sample_rows,
            oracle=oracle,
            efficiency_guidance=efficiency_guidance,
        )
        planner_messages_sha256 = sha256_text(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
        raw = ""
        while attempts < max(1, max_attempts):
            attempts += 1
            try:
                raw = planner_fn(messages)
                program = extract_code(raw)
                build_pipeline_from_code(program, loader)
                error = None
                break
            except PZInfrastructureError as exc:
                error = f"infrastructure: {exc}"
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempts < max(1, max_attempts):
                    messages.extend(
                        (
                            {"role": "assistant", "content": raw or program},
                            {
                                "role": "user",
                                "content": REPAIR_USER_TEMPLATE.format(
                                    traceback=traceback.format_exc(limit=4)
                                ),
                            },
                        )
                    )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if loader is not None:
            loader.close()

    return {
        "question_id": qid,
        "db": db_name,
        "query": str(query.get("query") or ""),
        "hint": str(query.get("hint") or ""),
        "required_columns": list(query.get("required_columns") or []),
        "oracle": bool(oracle),
        "efficiency_guidance": bool(efficiency_guidance),
        "program": program,
        "program_sha256": sha256_text(program),
        "planner_messages_sha256": planner_messages_sha256,
        "generation_error": error,
        "generation_attempts": attempts,
        "generation_wall_time_s": round(time.monotonic() - started, 3),
    }


def generate_with_query_retries(
    query: dict[str, Any],
    *,
    max_query_retries: int,
    query_retry_delay_s: float,
    **generation_kwargs: Any,
) -> dict[str, Any]:
    """Retry a failed query generation from its original prompt."""
    started = time.monotonic()
    errors: list[str] = []
    result: dict[str, Any] = {}
    attempts = max(1, int(max_query_retries) + 1)
    for query_attempt in range(1, attempts + 1):
        result = generate_one(query, **generation_kwargs)
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
        time.monotonic() - started, 3
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: generate fixed PZ programs from SWAN questions."
    )
    parser.add_argument(
        "--query_file",
        default=str((REPO_ROOT / "swan" / "evaluation.jsonl").resolve()),
    )
    parser.add_argument(
        "--out_file",
        help=(
            "Output JSONL. Defaults to generated_queries_fair.jsonl or "
            "generated_queries_oracle.jsonl beside this script."
        ),
    )
    parser.add_argument("--oracle", action="store_true")
    efficiency_group = parser.add_mutually_exclusive_group(required=True)
    efficiency_group.add_argument(
        "--efficiency-guidance",
        dest="efficiency_guidance",
        action="store_true",
    )
    efficiency_group.add_argument(
        "--no-efficiency-guidance",
        dest="efficiency_guidance",
        action="store_false",
    )
    parser.add_argument("--run_id", action="append", default=[])
    parser.add_argument("--run_db", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=env_int("E1_ROW_SAMPLE_SIZE", env_int("PZ_ROW_SAMPLE", 20)),
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=env_int(
            "E1_MAX_TRANSLATION_ATTEMPTS",
            env_int("PZ_MAX_TRANSLATION_ATTEMPTS", 3),
        ),
    )
    parser.add_argument(
        "--validation_row_cap",
        type=int,
        default=1,
        help="Rows loaded per referenced table while validating build_pipeline (default: 1).",
    )
    parser.add_argument(
        "--max_query_retries",
        type=int,
        default=env_int("E1_MAX_QUERY_GENERATION_RETRIES", 3),
        help="Fresh query-generation retries after final generation failure.",
    )
    parser.add_argument(
        "--query_retry_delay_s",
        type=float,
        default=float(env("E1_QUERY_GENERATION_RETRY_DELAY_S", "5")),
        help="Fixed delay between fresh query-generation retries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful records already present in the output JSONL and continue missing/failed questions.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Offline generation test using the live schema and a deterministic planner stub.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "oracle" if args.oracle else "fair"
    guidance = "efficiency" if args.efficiency_guidance else "no_efficiency"
    output = (
        Path(args.out_file).resolve()
        if args.out_file
        else THIS_DIR / f"generated_queries_{mode}_{guidance}.jsonl"
    )
    query_file = Path(args.query_file).resolve()
    queries = load_jsonl(query_file)
    if args.run_id:
        wanted = set(args.run_id)
        queries = [q for q in queries if str(q.get("question_id")) in wanted]
    elif args.run_db:
        wanted = set(args.run_db)
        queries = [q for q in queries if str(q.get("db")) in wanted]
    if args.limit > 0:
        queries = queries[: args.limit]

    planner_fn = mock_planner if args.mock else call_planner
    records_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output.is_file():
        for existing in load_jsonl(output):
            if (
                bool(existing.get("oracle")) == bool(args.oracle)
                and bool(existing.get("efficiency_guidance"))
                == bool(args.efficiency_guidance)
            ):
                records_by_id[str(existing.get("question_id") or "")] = existing
    resumed = sum(
        1
        for query in queries
        if (
            str(query.get("question_id") or "") in records_by_id
            and records_by_id[str(query.get("question_id") or "")].get(
                "generation_error"
            )
            is None
        )
    )
    if resumed:
        print(f"Resuming with {resumed} successful records already persisted.", flush=True)
    for index, query in enumerate(queries, 1):
        question_id = str(query.get("question_id") or "")
        record = records_by_id.get(question_id)
        if record is None or record.get("generation_error") is not None:
            record = generate_with_query_retries(
                query,
                max_query_retries=max(0, args.max_query_retries),
                query_retry_delay_s=max(0.0, args.query_retry_delay_s),
                oracle=bool(args.oracle),
                efficiency_guidance=bool(args.efficiency_guidance),
                sample_rows=args.sample_rows,
                max_attempts=args.max_attempts,
                validation_row_cap=args.validation_row_cap,
                planner_fn=planner_fn,
            )
            records_by_id[question_id] = record
        records = [
            records_by_id[str(source.get("question_id") or "")]
            for source in queries
            if str(source.get("question_id") or "") in records_by_id
        ]
        # Persist after every query so progress survives interruption and is
        # visible while a long generation run is still in progress.
        write_jsonl(output, records)
        print(
            f"[{index}/{len(queries)}] {record['question_id']}: "
            f"attempts={record['generation_attempts']} "
            f"error={record['generation_error']}",
            flush=True,
        )
    records = [
        records_by_id[str(query.get("question_id") or "")]
        for query in queries
        if str(query.get("question_id") or "") in records_by_id
    ]
    manifest = {
        "program_file": str(output),
        "program_file_sha256": sha256_file(output),
        "query_file": str(query_file),
        "query_file_sha256": sha256_file(query_file),
        "records": len(records),
        "oracle": bool(args.oracle),
        "efficiency_guidance": bool(args.efficiency_guidance),
        "mock": bool(args.mock),
        "planner_model": "stub" if args.mock else planner_model_id(),
        "planner_reasoning_effort": env(
            "PZ_PLANNER_REASONING_EFFORT", "high"
        ),
        "row_sample": int(args.sample_rows),
        "row_sample_max_chars": env_int("E1_ROW_SAMPLE_MAX_CHARS", 6000),
        "max_generation_attempts": int(args.max_attempts),
        "max_query_generation_retries": int(args.max_query_retries),
        "query_generation_retry_delay_s": float(args.query_retry_delay_s),
        "validation_row_cap": int(args.validation_row_cap),
        "prompt_file_sha256": {
            path.name: sha256_file(path)
            for path in sorted(PROMPTS_DIR.glob("*.md"))
        },
        "generation_wall_time_s_total": round(
            sum(float(record.get("generation_wall_time_s", 0) or 0) for record in records),
            3,
        ),
        "generation_time_is_execution_metric": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = sum(record["generation_error"] is not None for record in records)
    print(
        f"Wrote {len(records)} fixed programs to {output} ({failed} generation failures). "
        "generation_wall_time_s is stage-1 metadata and is never an execution metric.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
