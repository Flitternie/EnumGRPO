#!/usr/bin/env python3
"""Stage 2: execute pre-generated Palimpzest programs without query generation."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import (
    LIVE_USAGE_KEYS,
    PZInfrastructureError,
    PZTableLoader,
    build_config,
    build_pipeline_from_code,
    empty_usage,
    env,
    env_int,
    extract_usage,
    load_jsonl,
    read_usage_journal,
    recompute_cost,
    reset_usage,
    resolve_db_path,
    sanitize_qid,
    sha256_file,
    sha256_text,
    snapshot_usage,
)

SCRIPT = Path(__file__).resolve()


class QueryTimeout(BaseException):
    pass


def install_timeout(seconds: int):
    if seconds <= 0:
        return lambda: None

    def handler(signum, frame):
        raise QueryTimeout()

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))

    def restore() -> None:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous)

    return restore


def apply_mem_limit(mem_limit_gb: float) -> None:
    if mem_limit_gb <= 0:
        return
    try:
        import resource

        cap = int(mem_limit_gb * (1024**3))
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = cap if hard == resource.RLIM_INFINITY or cap < hard else hard
        resource.setrlimit(resource.RLIMIT_AS, (cap, new_hard))
    except Exception as exc:
        print(f"[worker] could not set RLIMIT_AS ({exc}); continuing", flush=True)


def base_usage(
    record: dict[str, Any], *, error: str | None, wall_time_s: float
) -> dict[str, Any]:
    return {
        "question_id": str(record.get("question_id") or ""),
        "db": str(record.get("db") or ""),
        "oracle": bool(record.get("oracle", False)),
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "llm_calls": 0,
        "wall_time_s": round(float(wall_time_s), 3),
        "cache_creation_tokens": 0,
        "embedding_input_tokens": 0,
        "cost_usd_ours": 0.0,
        "abacus_optimization_cost_usd_pz": 0.0,
        "operator_model": env("PZ_OP_MODEL") or env("LLMOP_MODEL"),
        "program_sha256": sha256_text(str(record.get("program") or "")),
        "n_pred_rows": 0,
        "timed_out": error == "timeout",
        "oom": error == "oom",
        "infra_failure": bool(error and error.startswith("infrastructure")),
        "error": error,
        "per_operator": [],
        "calls_with_usage": 0,
        "unmetered_failed_calls": 0,
        "excluded_infra_calls": 0,
        "in_flight_calls": 0,
        "usage_accounting": (
            "PZ optimization/operator calls and DB loading only; infrastructure retries "
            "excluded; stage-1 generation excluded"
        ),
    }


def write_result_files(
    run_dir: Path,
    record: dict[str, Any],
    frame,
    usage_json: dict[str, Any],
) -> None:
    safe_qid = sanitize_qid(str(record.get("question_id") or ""))
    csv_path = run_dir / f"{safe_qid}.csv"
    if frame is None:
        csv_path.write_text("", encoding="utf-8")
    else:
        try:
            frame.to_csv(csv_path, index=False)
        except Exception:
            csv_path.write_text("", encoding="utf-8")
            raise
    (run_dir / f"{safe_qid}.usage.json").write_text(
        json.dumps(usage_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_one(
    record: dict[str, Any],
    run_dir: Path,
    *,
    row_cap: int,
    query_timeout_s: int,
) -> dict[str, Any]:
    qid = str(record.get("question_id") or "").strip()
    safe_qid = sanitize_qid(qid)
    journal_path = run_dir / f"{safe_qid}.usage.partial.jsonl"
    journal_path.unlink(missing_ok=True)
    os.environ["PZ_USAGE_JOURNAL_PATH"] = str(journal_path)
    reset_usage()

    generation_error = record.get("generation_error")
    program = str(record.get("program") or "")
    if generation_error or not program.strip():
        error = f"generation_error: {generation_error or 'empty program'}"
        usage_json = base_usage(record, error=error, wall_time_s=0.0)
        usage_json["execution_skipped"] = True
        write_result_files(run_dir, record, None, usage_json)
        journal_path.unlink(missing_ok=True)
        return usage_json

    started = time.monotonic()
    with journal_path.open("a", encoding="utf-8") as journal:
        journal.write(
            json.dumps({"event": "execution_start", "monotonic_s": started}) + "\n"
        )
        journal.flush()
    loader: PZTableLoader | None = None
    frame = None
    raw_usage = empty_usage()
    error: str | None = None
    restore_timeout = install_timeout(query_timeout_s)
    try:
        # The measured clock begins before DB resolution/loading and ends after PZ returns.
        loader = PZTableLoader(
            resolve_db_path(str(record.get("db") or "")), row_cap=row_cap
        )
        dataset = build_pipeline_from_code(program, loader)
        collection = dataset.run(config=build_config())
        frame = collection.to_df()
        raw_usage = extract_usage(collection)
    except QueryTimeout:
        error = "timeout"
        frame = None
    except PZInfrastructureError as exc:
        error = f"infrastructure: {exc}"
        frame = None
    except MemoryError:
        error = "oom"
        frame = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        frame = None
        print(f"[worker] {qid}: {error}", flush=True)
        traceback.print_exc(limit=4)
    finally:
        restore_timeout()
        if loader is not None:
            loader.close()
    wall_time = (
        min(time.monotonic() - started, float(query_timeout_s))
        if error == "timeout" and query_timeout_s > 0
        else time.monotonic() - started
    )

    live = snapshot_usage()
    for key in LIVE_USAGE_KEYS:
        raw_usage[key] = int(live.get(key, 0))
    raw_usage["total_llm_calls"] = int(live.get("total_llm_calls", 0))
    usage_json = base_usage(record, error=error, wall_time_s=wall_time)
    usage_json.update(
        {
            "input_tokens": raw_usage["input_text_tokens"],
            "output_tokens": raw_usage["output_text_tokens"],
            "cached_tokens": raw_usage["cache_read_tokens"],
            "llm_calls": raw_usage["total_llm_calls"],
            "cache_creation_tokens": raw_usage["cache_creation_tokens"],
            "embedding_input_tokens": raw_usage["embedding_input_tokens"],
            "cost_usd_ours": round(recompute_cost(raw_usage), 6),
            "abacus_optimization_cost_usd_pz": raw_usage[
                "optimization_cost_usd"
            ],
            "n_pred_rows": int(len(frame)) if frame is not None else 0,
            "timed_out": error == "timeout",
            "oom": error == "oom",
            "infra_failure": bool(error and error.startswith("infrastructure")),
            "per_operator": raw_usage["per_operator"],
            "calls_with_usage": int(live.get("calls_with_usage", 0)),
            "unmetered_failed_calls": int(
                live.get("unmetered_failed_calls", 0)
            ),
            "excluded_infra_calls": int(live.get("excluded_infra_calls", 0)),
            "in_flight_calls": int(live.get("in_flight_calls", 0)),
            "execution_skipped": False,
        }
    )
    try:
        write_result_files(run_dir, record, frame, usage_json)
    except Exception as exc:
        usage_json["error"] = f"output_error: {type(exc).__name__}: {exc}"
        write_result_files(run_dir, record, None, usage_json)
    journal_path.unlink(missing_ok=True)
    print(
        f"{qid}: rows={usage_json['n_pred_rows']} in={usage_json['input_tokens']} "
        f"out={usage_json['output_tokens']} calls={usage_json['llm_calls']} "
        f"wall={usage_json['wall_time_s']:.3f}s error={usage_json['error']}",
        flush=True,
    )
    return usage_json


def recover_hard_failure(
    record: dict[str, Any],
    run_dir: Path,
    *,
    error: str,
    wall_time_s: float,
) -> None:
    qid = str(record.get("question_id") or "").strip()
    safe_qid = sanitize_qid(qid)
    journal_path = run_dir / f"{safe_qid}.usage.partial.jsonl"
    partial = read_usage_journal(journal_path)
    usage_json = base_usage(record, error=error, wall_time_s=wall_time_s)
    usage_json.update(
        {
            "input_tokens": partial["input_text_tokens"],
            "output_tokens": partial["output_text_tokens"],
            "cached_tokens": partial["cache_read_tokens"],
            "llm_calls": partial["total_llm_calls"],
            "cache_creation_tokens": partial["cache_creation_tokens"],
            "embedding_input_tokens": partial["embedding_input_tokens"],
            "cost_usd_ours": round(recompute_cost(partial), 6),
            "calls_with_usage": partial["calls_with_usage"],
            "unmetered_failed_calls": partial["unmetered_failed_calls"],
            "excluded_infra_calls": partial["excluded_infra_calls"],
            "in_flight_calls": partial["in_flight_calls"],
            "execution_skipped": False,
        }
    )
    write_result_files(run_dir, record, None, usage_json)
    journal_path.unlink(missing_ok=True)


def recovered_execution_wall(journal_path: Path, fallback: float) -> float:
    """Recover the measured child interval without counting process startup."""
    try:
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "execution_start":
                return max(0.0, time.monotonic() - float(event["monotonic_s"]))
    except Exception:
        pass
    return fallback


def select_records(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    for record in records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            raise ValueError("program record has an empty question_id")
        if qid in seen:
            raise ValueError(f"duplicate question_id in program file: {qid}")
        seen.add(qid)
        if not isinstance(record.get("program", ""), str):
            raise ValueError(f"{qid}: program must be a string")
        program_hash = sha256_text(str(record.get("program") or ""))
        expected_hash = str(record.get("program_sha256") or "")
        if expected_hash and expected_hash != program_hash:
            raise ValueError(f"{qid}: program_sha256 does not match program content")
    if args.run_id:
        wanted = set(args.run_id)
        records = [record for record in records if record["question_id"] in wanted]
    elif args.run_db:
        wanted = set(args.run_db)
        records = [record for record in records if record.get("db") in wanted]
    if args.limit > 0:
        records = records[: args.limit]
    return records


def run_child(
    record: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    hard_timeout: int,
) -> None:
    qid = str(record["question_id"])
    safe_qid = sanitize_qid(qid)
    usage_path = run_dir / f"{safe_qid}.usage.json"
    journal_path = run_dir / f"{safe_qid}.usage.partial.jsonl"
    usage_path.unlink(missing_ok=True)
    journal_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--program_file",
        str(Path(args.program_file).resolve()),
        "--run_dir",
        str(run_dir),
        "--run_id",
        qid,
        "--row_cap",
        str(args.row_cap),
        "--query_timeout_s",
        str(args.query_timeout_s),
        "--mem_limit_gb",
        str(args.mem_limit_gb),
    ]
    if args.mock:
        command.append("--mock")
    started = time.monotonic()
    status = "ok"
    return_code = 0
    stderr_tail = ""
    try:
        completed = subprocess.run(
            command, timeout=hard_timeout, capture_output=True, text=True
        )
        return_code = completed.returncode
        if return_code != 0:
            status = f"child_exit_{return_code}"
        output_lines = (completed.stdout or "").strip().splitlines()
        if output_lines:
            print(f"  [{qid}] {output_lines[-1]}", flush=True)
        if completed.stderr and return_code != 0:
            stderr_tail = completed.stderr.strip().splitlines()[-1]
            print(
                f"  [{qid}] stderr: {stderr_tail}",
                flush=True,
            )
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception as exc:
        status = f"orchestrator_error: {exc}"
    elapsed = time.monotonic() - started
    if not usage_path.is_file():
        if status == "timeout":
            error = "timeout"
        elif return_code < 0 and -return_code in {signal.SIGKILL, signal.SIGSEGV}:
            error = "oom"
        else:
            error = status
            if stderr_tail:
                error = f"{error}: {stderr_tail}"
        recover_hard_failure(
            record,
            run_dir,
            error=error,
            wall_time_s=recovered_execution_wall(journal_path, elapsed),
        )
        print(f"  [{qid}] recovered hard failure: {error}", flush=True)
    journal_path.unlink(missing_ok=True)


def write_manifest(run_dir: Path) -> None:
    records: list[str] = []
    for usage_path in sorted(run_dir.glob("*.usage.json")):
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
            usage.pop("per_operator", None)
            records.append(json.dumps(usage, ensure_ascii=False))
        except Exception:
            pass
    path = run_dir / "pz_manifest.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        ("\n".join(records) + "\n") if records else "", encoding="utf-8"
    )
    temporary.replace(path)


def read_usage_file(
    run_dir: Path, record: dict[str, Any]
) -> dict[str, Any] | None:
    qid = sanitize_qid(str(record.get("question_id") or ""))
    path = run_dir / f"{qid}.usage.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def load_resumable_usage(
    run_dir: Path,
    record: dict[str, Any],
    *,
    retry_infra_only: bool = False,
) -> dict[str, Any] | None:
    """Return a completed measured result, or None when the task must rerun."""
    usage = read_usage_file(run_dir, record)
    if usage is None:
        return None
    error = str(usage.get("error") or "")
    if usage.get("program_sha256") != sha256_text(
        str(record.get("program") or "")
    ):
        return None
    if retry_infra_only:
        return (
            None
            if usage.get("infra_failure") or error.startswith("infrastructure")
            else usage
        )
    if (
        usage.get("infra_failure")
        or error.startswith("orchestrator_error")
        or error.startswith("output_error")
        or error.startswith("child_exit_")
    ):
        return None
    return usage


def task_result_record(
    record: dict[str, Any],
    usage: dict[str, Any],
    *,
    index: int,
    run_dir: Path,
) -> dict[str, Any]:
    error = str(usage.get("error") or "")
    ok = not error
    timed_out = error == "timeout"
    question_id = str(record.get("question_id") or "").strip()
    db = str(record.get("db") or "").strip()
    try:
        db_path = resolve_db_path(db)
    except Exception:
        db_path = Path(env("DB_FILES_DIR") or "") / f"{db}.duckdb"
    return {
        "ok": ok,
        "returncode": 0 if ok else (None if timed_out else 1),
        "elapsed_s": float(usage.get("wall_time_s", 0.0) or 0.0),
        "stdout": "",
        "stderr": error,
        "execution_status": (
            "infrastructure_excluded"
            if usage.get("infra_failure")
            else ("timeout" if timed_out else ("ok" if ok else "error"))
        ),
        "job": {
            "idx": index,
            "question_id": question_id,
            "db": db,
            "db_path": str(db_path),
            "output_path": str(
                run_dir / f"{sanitize_qid(question_id)}.csv"
            ),
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


def run_orchestrator(
    records: list[dict[str, Any]], run_dir: Path, args: argparse.Namespace
) -> int:
    program_file = Path(args.program_file).resolve()
    manifest = {
        "program_file": str(program_file),
        "program_file_sha256": sha256_file(program_file),
        "program_file_records": len(load_jsonl(program_file)),
        "selected_records": len(records),
        "programs_regenerated": False,
        "program_hashes": {
            str(record["question_id"]): sha256_text(
                str(record.get("program") or "")
            )
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
    pending_records: list[tuple[int, dict[str, Any]]] = []
    task_results: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        existing_usage = (
            load_resumable_usage(
                run_dir,
                record,
                retry_infra_only=args.retry_infra_only,
            )
            if args.resume
            else None
        )
        if existing_usage is not None:
            print(
                f"  [{record['question_id']}] resume: keeping completed result",
                flush=True,
            )
            task_results.append(
                task_result_record(
                    record,
                    existing_usage,
                    index=record_index,
                    run_dir=run_dir,
                )
            )
        else:
            pending_records.append((record_index, record))
    write_manifest(run_dir)
    write_task_logs(run_dir, task_results)
    if not pending_records:
        print(f"All {len(records)} tasks already complete in {run_dir}", flush=True)
        return 0
    concurrency = max(1, int(args.concurrency))
    hard_timeout = max(1, int(args.query_timeout_s)) + 180
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                run_child, record, run_dir, args, hard_timeout
            ): (record_index, record)
            for record_index, record in pending_records
        }
        for index, future in enumerate(as_completed(futures), 1):
            record_index, record = futures[future]
            qid = str(record["question_id"])
            try:
                future.result()
            except Exception as exc:
                print(f"  [{qid}] orchestrator task error: {exc}", flush=True)
                usage = base_usage(
                    record,
                    error=f"orchestrator_error: {exc}",
                    wall_time_s=0.0,
                )
                write_result_files(run_dir, record, None, usage)
            usage = read_usage_file(run_dir, record)
            if usage is None:
                usage = base_usage(
                    record,
                    error="orchestrator_error: missing usage record",
                    wall_time_s=0.0,
                )
                write_result_files(run_dir, record, None, usage)
            task_results.append(
                task_result_record(
                    record,
                    usage,
                    index=record_index,
                    run_dir=run_dir,
                )
            )
            write_manifest(run_dir)
            write_task_logs(run_dir, task_results)
            print(f"[{index}/{len(pending_records)}] finished {qid}", flush=True)
    write_manifest(run_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: repeatedly execute fixed PZ program JSONL records."
    )
    parser.add_argument("--program_file", required=True)
    parser.add_argument(
        "--out_dir",
        help="Parent output directory; creates run_r1 ... run_rK.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--run_id", action="append", default=[])
    parser.add_argument("--run_db", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--row_cap",
        type=int,
        default=env_int("PZ_TABLE_ROW_CAP", 0),
        help="Debug-only table row cap; use 0 for benchmark runs.",
    )
    parser.add_argument(
        "--query_timeout_s",
        type=int,
        default=env_int("QUERY_TIMEOUT_S", 1800),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=env_int("QUERY_CONCURRENCY", 4),
    )
    parser.add_argument(
        "--mem_limit_gb",
        type=float,
        default=float(env("E1_MEM_LIMIT_GB") or env("PZ_MEM_LIMIT_GB") or "48"),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Offline smoke mode for a relational-only mock program.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed tasks with matching program hashes; rerun infrastructure or invalid records.",
    )
    parser.add_argument(
        "--retry_infra_only",
        action="store_true",
        help=(
            "With --resume, rerun only model/API infrastructure failures while "
            "preserving measured timeouts, OOMs, code errors, and child exits."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run_dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_runs < 1:
        raise ValueError("--num_runs must be at least 1")
    if args.retry_infra_only and not args.resume:
        raise ValueError("--retry_infra_only requires --resume")
    if not args.mock and not (env("PZ_OP_MODEL") or env("LLMOP_MODEL")):
        raise RuntimeError(
            "Operator model is not configured. Set LLMOP_MODEL in the repository "
            ".env or PZ_OP_MODEL in baseline/palimpzest/.env."
        )
    records = select_records(
        load_jsonl(Path(args.program_file).resolve()), args
    )
    if args.mock:
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy-offline")

    if args.worker:
        if not args.run_dir:
            raise ValueError("internal --worker requires --run_dir")
        apply_mem_limit(args.mem_limit_gb)
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            run_one(
                record,
                run_dir,
                row_cap=args.row_cap,
                query_timeout_s=args.query_timeout_s,
            )
        return 0

    if not args.out_dir:
        raise ValueError("--out_dir is required")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Loaded {len(records)} fixed programs; executing {args.num_runs} run(s).",
        flush=True,
    )
    for run_number in range(1, args.num_runs + 1):
        run_dir = out_dir / f"run_r{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== run_r{run_number} ===", flush=True)
        result = run_orchestrator(records, run_dir, args)
        if result:
            return result
    print(f"\nDone. Outputs: {out_dir}/run_r1 ... run_r{args.num_runs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
