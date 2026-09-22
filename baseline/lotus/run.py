#!/usr/bin/env python3
"""Stage 2: execute pre-generated LOTUS programs without invoking a planner."""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from common import (
    PROMPTS_DIR,
    REPO_ROOT,
    SYSTEM_DIR,
    clean_result_dataframe,
    load_db_tables,
    load_jsonl,
    resolve_duckdb_path,
    resolve_under_repo,
    select_records,
    sha256_file,
    sha256_text,
)

if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass


class InfrastructureError(RuntimeError):
    """Operator model/API failure after infrastructure retries are exhausted."""


class QueryTimeout(BaseException):
    """Per-query wall-clock timeout that generated code cannot catch as Exception."""


PRICE = {
    "input": 1.00 / 1_000_000,
    "cache_read": 0.10 / 1_000_000,
    "cache_write": 1.25 / 1_000_000,
    "output": 5.00 / 1_000_000,
}


def timeout_handler(signum, frame):  # pragma: no cover - signal-driven
    raise QueryTimeout()


def is_bedrock(model: str) -> bool:
    return str(model or "").lower().startswith("bedrock/")


def inject_operator_directive(messages: Any) -> Any:
    directive = " " + (
        PROMPTS_DIR / "operator_system_suffix.md"
    ).read_text(encoding="utf-8").strip()
    try:
        output = []
        for conversation in messages:
            copied = [dict(message) for message in conversation]
            for message in copied:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + directive
                    break
            else:
                copied.insert(0, {"role": "system", "content": directive.strip()})
            output.append(copied)
        return output
    except Exception:
        return messages


class StubLM:
    """Deterministic offline LOTUS operator LM."""

    def __init__(self):
        from lotus.cache import CacheFactory
        from lotus.types import LMStats

        self.model = "stub/operator"
        self.stats = LMStats()
        self.cache = CacheFactory.create_default_cache()
        self.llm_calls = 0
        self.excluded_infra_calls = 0
        self.max_tokens = 512
        self.max_ctx_len = 128000
        self.max_batch_size = 1

    def __call__(self, messages, show_progress_bar=False, progress_bar_desc="", **kwargs):
        from lotus.types import LMOutput

        count = len(messages)
        self.llm_calls += count
        for usage in (self.stats.virtual_usage, self.stats.physical_usage):
            usage.prompt_tokens += 10 * count
            usage.completion_tokens += 2 * count
            usage.total_tokens += 12 * count
        logprobs = [[] for _ in range(count)] if kwargs.get("logprobs") else None
        return LMOutput(outputs=["True" for _ in range(count)], logprobs=logprobs)

    def count_tokens(self, messages):
        return 0

    def get_model_name(self) -> str:
        return "stub-operator"

    def is_deepseek(self) -> bool:
        return False

    def is_reasoning_model(self) -> bool:
        return False

    def reset_stats(self):
        from lotus.types import LMStats

        self.stats = LMStats()

    def reset_cache(self, max_size=None):
        self.cache.reset(max_size)

    def print_total_usage(self):
        pass


def build_operator_lm(*, dry_run: bool):
    from lotus.models import LM

    if dry_run:
        return StubLM()

    model = (os.getenv("LLMOP_MODEL") or "").strip()
    if not model:
        raise RuntimeError("Operator model not configured (set LLMOP_MODEL).")
    kwargs: dict[str, Any] = {"model": model}
    if not is_bedrock(model):
        api_key = (os.getenv("LLMOP_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("LLMOP_API_KEY required for a non-Bedrock operator model.")
        kwargs["api_key"] = api_key
        base_url = (os.getenv("LLMOP_BASE_URL") or "").strip()
        if base_url:
            kwargs["base_url"] = base_url
    concurrency = int((os.getenv("LLMOP_CONCURRENCY") or "4").strip() or "4")

    class CountingLM(LM):
        def __init__(self, *args, **model_kwargs):
            super().__init__(*args, **model_kwargs)
            self.llm_calls = 0
            self.excluded_infra_calls = 0

        def __call__(self, messages, *args, **call_kwargs):
            messages = inject_operator_directive(messages)
            max_infra_attempts = max(
                1, int((os.getenv("E1_MAX_INFRA_ATTEMPTS") or "3").strip() or "3")
            )
            for attempt in range(1, max_infra_attempts + 1):
                stats_before = copy.deepcopy(self.stats)
                try:
                    output = super().__call__(messages, *args, **call_kwargs)
                    self.llm_calls += len(messages)
                    return output
                except Exception as exc:
                    self.stats = stats_before
                    self.excluded_infra_calls += len(messages)
                    if attempt >= max_infra_attempts:
                        raise InfrastructureError(
                            "operator model/API call failed after "
                            f"{max_infra_attempts} attempts: {exc}"
                        ) from exc
                    time.sleep(min(2 ** (attempt - 1), 8))
            raise AssertionError("unreachable")

    lm = CountingLM(max_batch_size=max(1, concurrency), **kwargs)
    lm.kwargs.pop("temperature", None)
    return lm


def usage_snapshot(lm: Any) -> dict[str, int | float]:
    convention = (os.getenv("E1_TOKEN_CONVENTION") or "physical").strip().lower()
    usage = lm.stats.physical_usage if convention == "physical" else lm.stats.virtual_usage
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cached_tokens": int(getattr(usage, "cached_prompt_tokens", 0) or 0),
        "cache_creation_tokens": int(getattr(usage, "cache_creation_tokens", 0) or 0),
        "llm_calls": int(getattr(lm, "llm_calls", 0) or 0),
        "model_cache_hits": int(getattr(lm.stats, "cache_hits", 0) or 0),
        "operator_cache_hits": int(
            getattr(lm.stats, "operator_cache_hits", 0) or 0
        ),
    }


def recompute_cost(usage: dict[str, int | float]) -> float:
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    cache_read = int(usage.get("cached_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_tokens", 0) or 0)
    non_cached_input = max(0, prompt_tokens - cache_read - cache_write)
    return (
        non_cached_input * PRICE["input"]
        + cache_read * PRICE["cache_read"]
        + cache_write * PRICE["cache_write"]
        + int(usage.get("output_tokens", 0) or 0) * PRICE["output"]
    )


def usage_delta(
    before: dict[str, int | float], after: dict[str, int | float]
) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key in set(before) | set(after):
        delta = max(0.0, float(after.get(key, 0)) - float(before.get(key, 0)))
        result[key] = int(delta)
    return result


def validate_program_records(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        question_id = str(record.get("question_id") or "").strip()
        db = str(record.get("db") or "").strip()
        if not question_id or not db or "program" not in record:
            raise ValueError(
                f"program record {index} must contain question_id, db, and program"
            )
        if question_id in seen:
            raise ValueError(f"duplicate question_id in program file: {question_id}")
        seen.add(question_id)


def write_result_csv(
    *,
    result: Any,
    required_columns: list[str],
    csv_path: Path,
    pandas_module: Any,
) -> int:
    if result is None:
        dataframe = pandas_module.DataFrame()
    elif isinstance(result, pandas_module.DataFrame):
        dataframe = result
    else:
        dataframe = pandas_module.DataFrame(result)
    dataframe = clean_result_dataframe(dataframe)
    if required_columns and len(dataframe.columns):
        lowercase_columns = {str(column).lower(): column for column in dataframe.columns}
        selected = [
            lowercase_columns[column.lower()]
            for column in required_columns
            if column.lower() in lowercase_columns
        ]
        if selected:
            dataframe = dataframe[selected]
    try:
        dataframe.to_csv(csv_path, index=False)
    except Exception:
        dataframe = pandas_module.DataFrame()
        dataframe.to_csv(csv_path, index=False)
    return len(dataframe)


def run_one(
    record: dict[str, Any],
    *,
    lm: Any,
    db_dir: Path,
    out_dir: Path,
    query_timeout_s: int,
    run_index: int,
) -> dict[str, Any]:
    import lotus
    import pandas as pd

    question_id = str(record.get("question_id") or "").strip()
    db = str(record.get("db") or "").strip()
    query = str(record.get("query") or "")
    program = str(record.get("program") or "")
    required_columns = list(record.get("required_columns") or [])
    program_hash = sha256_text(program)
    expected_hash = str(record.get("program_sha256") or "")
    if expected_hash and expected_hash != program_hash:
        raise ValueError(f"[{question_id}] program_sha256 does not match program content")

    print(f"[r{run_index}:{question_id}] db={db} :: {query[:80]}", flush=True)
    lm.reset_stats()
    lm.reset_cache()
    lm.llm_calls = 0
    lm.excluded_infra_calls = 0
    stats_before = copy.deepcopy(lm.stats)
    usage_before = usage_snapshot(lm)
    calls_before = 0
    result = None
    error: str | None = None
    status = "ok"
    use_timer = (
        query_timeout_s > 0
        and threading.current_thread() is threading.main_thread()
    )
    previous_handler = None
    started = time.perf_counter()
    if use_timer:
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(query_timeout_s))

    try:
        tables = load_db_tables(resolve_duckdb_path(db, db_dir))
        if not program.strip():
            raise ValueError(
                "fixed program is empty"
                + (
                    f" (generation_error={record.get('generation_error')})"
                    if record.get("generation_error")
                    else ""
                )
            )
        namespace: dict[str, Any] = {"pd": pd, "lotus": lotus, "tables": tables}
        namespace.update(tables)
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(compile(program, f"<lotus-program:{question_id}>", "exec"), namespace)
        result = namespace.get("result_df")
        if result is None:
            raise ValueError("fixed program did not assign `result_df`")
    except QueryTimeout:
        error = "timeout"
        status = "timeout"
        result = None
        print(f"[r{run_index}:{question_id}] TIMEOUT after {query_timeout_s}s", flush=True)
    except InfrastructureError as exc:
        error = f"infrastructure: {exc}"
        status = "infrastructure_excluded"
        result = None
        lm.stats = stats_before
        lm.llm_calls = calls_before
        print(f"[r{run_index}:{question_id}] INFRASTRUCTURE FAILURE: {exc}", flush=True)
    except Exception:
        error = "code_error"
        status = "code_error"
        result = None
        print(
            f"[r{run_index}:{question_id}] CODE ERROR:\n"
            f"{traceback.format_exc(limit=6)}",
            flush=True,
        )
    finally:
        if use_timer:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    wall_time = time.perf_counter() - started
    if error == "timeout":
        wall_time = min(wall_time, float(query_timeout_s))
    out_dir.mkdir(parents=True, exist_ok=True)
    row_count = write_result_csv(
        result=result,
        required_columns=required_columns,
        csv_path=out_dir / f"{question_id}.csv",
        pandas_module=pd,
    )
    measured_usage = usage_delta(usage_before, usage_snapshot(lm))
    measured_usage["cost_usd"] = round(recompute_cost(measured_usage), 10)
    usage = {
        **measured_usage,
        "question_id": question_id,
        "db": db,
        "run_index": run_index,
        "wall_time_s": round(wall_time, 3),
        "error": error,
        "status": status,
        "infra_failure": status == "infrastructure_excluded",
        "excluded_infra_calls": int(getattr(lm, "excluded_infra_calls", 0) or 0),
        "program_sha256": program_hash,
        "program_generation_error": record.get("generation_error"),
        "usage_scope": "DuckDB load walltime plus LOTUS operator execution; planner excluded",
        "usage_accounting": (
            "normal code errors and timeouts included; operator model/API "
            "infrastructure retries excluded"
        ),
    }
    (out_dir / f"{question_id}.usage.json").write_text(
        json.dumps(usage, indent=2) + "\n",
        encoding="utf-8",
    )
    lm.reset_cache()
    print(
        f"[r{run_index}:{question_id}] done rows={row_count} error={error} "
        f"in={usage['input_tokens']} out={usage['output_tokens']} "
        f"cost=${usage['cost_usd']:.6f} t={usage['wall_time_s']}s",
        flush=True,
    )
    return usage


def run_summary(usages: list[dict[str, Any]], run_index: int) -> dict[str, Any]:
    summed_fields = [
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cache_creation_tokens",
        "llm_calls",
        "model_cache_hits",
        "operator_cache_hits",
        "cost_usd",
        "wall_time_s",
        "excluded_infra_calls",
    ]
    totals = {
        field: sum(float(usage.get(field, 0) or 0) for usage in usages)
        for field in summed_fields
    }
    for field in summed_fields:
        if field not in {"cost_usd", "wall_time_s"}:
            totals[field] = int(totals[field])
    totals["cost_usd"] = round(float(totals["cost_usd"]), 10)
    totals["wall_time_s"] = round(float(totals["wall_time_s"]), 3)
    return {
        "run_index": run_index,
        "queries": len(usages),
        "errors": sum(1 for usage in usages if usage.get("error")),
        "totals": totals,
        "queries_usage": usages,
    }


def load_resumable_usage(
    run_dir: Path, record: dict[str, Any], run_index: int
) -> dict[str, Any] | None:
    """Return a completed measured result, or None when the task must rerun."""
    question_id = str(record.get("question_id") or "").strip()
    path = run_dir / f"{question_id}.usage.json"
    if not path.is_file():
        return None
    try:
        usage = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if usage.get("infra_failure") or usage.get("status") == "fatal":
        return None
    expected_hash = sha256_text(str(record.get("program") or ""))
    if usage.get("program_sha256") != expected_hash:
        return None
    if int(usage.get("run_index", 0) or 0) != int(run_index):
        return None
    return usage


def write_run_usage(
    run_dir: Path, usages: list[dict[str, Any]], run_index: int
) -> None:
    """Atomically checkpoint aggregate usage after every completed task."""
    path = run_dir / "usage.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run_summary(usages, run_index), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_result_record(
    record: dict[str, Any],
    usage: dict[str, Any],
    *,
    index: int,
    run_dir: Path,
    db_dir: Path,
) -> dict[str, Any]:
    error = str(usage.get("error") or "")
    ok = not error
    timed_out = error == "timeout"
    question_id = str(record.get("question_id") or "").strip()
    db = str(record.get("db") or "").strip()
    try:
        db_path = resolve_duckdb_path(db, db_dir)
    except Exception:
        db_path = db_dir / f"{db}.duckdb"
    return {
        "ok": ok,
        "returncode": 0 if ok else (None if timed_out else 1),
        "elapsed_s": float(usage.get("wall_time_s", 0.0) or 0.0),
        "stdout": "",
        "stderr": error,
        "execution_status": str(usage.get("status") or ""),
        "job": {
            "idx": index,
            "question_id": question_id,
            "db": db,
            "db_path": str(db_path),
            "output_path": str(run_dir / f"{question_id}.csv"),
            "run_dir": str(run_dir),
        },
    }


def write_task_logs(run_dir: Path, results: list[dict[str, Any]]) -> None:
    """Atomically checkpoint agentic-compatible task logs."""
    for filename, rows in (
        ("results.jsonl", results),
        ("failures.jsonl", [result for result in results if not result["ok"]]),
    ):
        path = run_dir / filename
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)


def parse_args() -> argparse.Namespace:
    db_default = os.getenv("DB_FILES_DIR") or str(REPO_ROOT / "swan" / "database")
    timeout_default = int(
        (os.getenv("QUERY_TIMEOUT_S") or "1800").strip() or "1800"
    )
    parser = argparse.ArgumentParser(
        description="Execute fixed LOTUS SWAN programs without planner calls."
    )
    parser.add_argument("--program_file", required=True, help="Stage-1 generated program JSONL.")
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "runs" / "lotus"), help="Root output directory; run_r1...run_rK are created below it.")
    parser.add_argument("--db_dir", default=db_default, help="Directory containing SWAN DuckDB files.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of independent executions of each fixed program.")
    parser.add_argument("--query_timeout_s", type=int, default=timeout_default, help="DB-load plus operator-execution timeout per question; defaults to QUERY_TIMEOUT_S, or 1800 when unset; 0 disables.")
    parser.add_argument("--limit", type=int, default=0, help="Execute only the first N matched programs.")
    parser.add_argument("--run_id", action="append", default=[], help="Only these question IDs; repeatable.")
    parser.add_argument("--run_db", action="append", default=[], help="Only these database names; repeatable.")
    parser.add_argument("--dry_run", action="store_true", help="Use the offline deterministic operator stub; never access the network.")
    parser.add_argument("--resume", action="store_true", help="Skip completed tasks with matching program hashes; rerun infrastructure or invalid records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num_runs must be at least 1")

    import lotus

    lotus.settings.enable_cache = True
    program_file = resolve_under_repo(args.program_file)
    all_records = load_jsonl(program_file)
    validate_program_records(all_records)
    records = select_records(
        all_records,
        run_ids=args.run_id,
        run_dbs=args.run_db,
        limit=args.limit,
    )
    program_file_hash = sha256_file(program_file)
    out_root = resolve_under_repo(args.out_dir)
    db_dir = resolve_under_repo(args.db_dir)
    print(
        f"LOTUS execution stage: {len(records)} fixed programs x {args.num_runs} runs | "
        f"program_file_sha256={program_file_hash} | dry_run={args.dry_run}",
        flush=True,
    )

    for run_index in range(1, args.num_runs + 1):
        run_dir = out_root / f"run_r{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_index": run_index,
            "program_file": str(program_file),
            "program_file_sha256": program_file_hash,
            "program_file_records": len(all_records),
            "selected_records": len(records),
            "programs_regenerated": False,
            "program_hashes": {
                str(record["question_id"]): sha256_text(str(record.get("program") or ""))
                for record in records
            },
        }
        manifest_path = run_dir / "program_manifest.json"
        if args.resume and manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing_manifest.get("program_file_sha256")
                != manifest["program_file_sha256"]
                or existing_manifest.get("program_hashes")
                != manifest["program_hashes"]
            ):
                raise ValueError(
                    f"{run_dir}: existing program manifest does not match --program_file"
                )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        lm = build_operator_lm(dry_run=bool(args.dry_run))
        lotus.settings.configure(lm=lm)
        usages: list[dict[str, Any]] = []
        task_results: list[dict[str, Any]] = []
        for record_index, record in enumerate(records):
            if args.resume:
                existing_usage = load_resumable_usage(
                    run_dir, record, run_index
                )
                if existing_usage is not None:
                    usages.append(existing_usage)
                    write_run_usage(run_dir, usages, run_index)
                    task_results.append(
                        task_result_record(
                            record,
                            existing_usage,
                            index=record_index,
                            run_dir=run_dir,
                            db_dir=db_dir,
                        )
                    )
                    write_task_logs(run_dir, task_results)
                    print(
                        f"[r{run_index}:{record['question_id']}] resume: "
                        "keeping completed result",
                        flush=True,
                    )
                    continue
            try:
                usages.append(
                    run_one(
                        record,
                        lm=lm,
                        db_dir=db_dir,
                        out_dir=run_dir,
                        query_timeout_s=max(0, args.query_timeout_s),
                        run_index=run_index,
                    )
                )
            except Exception as exc:
                question_id = str(record.get("question_id") or "").strip()
                print(f"[r{run_index}:{question_id}] FATAL: {exc}", flush=True)
                import pandas as pd

                pd.DataFrame().to_csv(run_dir / f"{question_id}.csv", index=False)
                fatal_usage = {
                    "question_id": question_id,
                    "db": str(record.get("db") or ""),
                    "run_index": run_index,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                    "llm_calls": 0,
                    "model_cache_hits": 0,
                    "operator_cache_hits": 0,
                    "cost_usd": 0.0,
                    "wall_time_s": 0.0,
                    "error": f"fatal: {type(exc).__name__}: {exc}",
                    "status": "fatal",
                    "program_sha256": sha256_text(str(record.get("program") or "")),
                }
                (run_dir / f"{question_id}.usage.json").write_text(
                    json.dumps(fatal_usage, indent=2) + "\n",
                    encoding="utf-8",
                )
                usages.append(fatal_usage)
                try:
                    lm.reset_cache()
                except Exception:
                    pass
            write_run_usage(run_dir, usages, run_index)
            task_results.append(
                task_result_record(
                    record,
                    usages[-1],
                    index=record_index,
                    run_dir=run_dir,
                    db_dir=db_dir,
                )
            )
            write_task_logs(run_dir, task_results)
        print(f"Run {run_index} done. Outputs in {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
