#!/usr/bin/env python3
"""Batch runner: BlendSQL agent over swan/swan.jsonl.

Reads the `query` field (natural language) from each JSONL entry and spawns
`python baseline/blendsql_agent.py db ...` as a subprocess — the agent
autonomously generates BlendSQL (or pure SQL) to answer the question.

Follows the same pattern as run_swan_main.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# Reuse helpers from run_swan_main.py.
from run_swan_main import (  # noqa: E402
    _csv_filename_from_question_id,
    _norm_db_key,
    _read_jobs,
    _repo_root,
    _resolve_duckdb_path,
    _run_folder_from_question_id,
    _run_one_script,
    _safe_slug,
    _timestamp_id,
)
from utils.snapshot import save_config_snapshot as _save_config_snapshot



def _clear_blendsql_cache() -> None:
    """Delete the BlendSQL diskcache before a run so token usage is fully measured."""
    import shutil
    import platformdirs

    cache_dir = Path(platformdirs.user_cache_dir("blendsql"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[cache] Cleared BlendSQL disk cache at {cache_dir}", file=sys.stderr)
    else:
        print(f"[cache] No BlendSQL disk cache found at {cache_dir} (nothing to clear)", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = _repo_root()
    p = argparse.ArgumentParser(description="Run BlendSQL agent on swan/swan.jsonl with multiprocessing.")
    p.add_argument("--query_file", default=str((repo_root / "swan" / "swan.jsonl").resolve()),
                   help="Path to the input JSONL file (default: swan/swan.jsonl).")
    p.add_argument(
        "--db_dir",
        default=None,
        help="Directory containing .duckdb files. Falls back to DB_FILES_DIR env var.",
    )
    _default_out_dir = str((repo_root / "swan" / "logs" / f"blendsql_agent_run_{_timestamp_id()}").resolve())
    p.add_argument(
        "--out_dir",
        default=_default_out_dir,
        help="Output root dir for artifacts + per-query agent run logs (default: swan/logs/blendsql_agent_run_<timestamp>).",
    )
    p.add_argument(
        "--agent_model",
        default=None,
        help="Sets AGENT_MODEL for the agent subprocesses (optional if already in .env or environment).",
    )
    p.add_argument(
        "--llmop_model",
        default=None,
        help="Sets LLMOP_MODEL for BlendSQL ingredient model (optional if already in .env or environment).",
    )
    p.add_argument("--concurrency", type=int, default=None,
                   help="Number of parallel workers (overrides QUERY_CONCURRENCY).")
    p.add_argument(
        "--run_id",
        action="append",
        default=[],
        help="Run only these question_id values (repeatable). Exact match.",
    )
    p.add_argument(
        "--run_db",
        action="append",
        default=[],
        help="Run only queries belonging to these db values (repeatable).",
    )
    p.add_argument("--timeout_s", type=int, default=None, help="Per-job timeout in seconds (overrides QUERY_TIMEOUT_S).")
    args = p.parse_args(argv)

    query_file = Path(args.query_file).resolve()
    # Resolve timeout: CLI arg > QUERY_TIMEOUT_S env var.
    _timeout_env = os.environ.get("QUERY_TIMEOUT_S")
    if args.timeout_s is not None:
        timeout_s = args.timeout_s
    elif _timeout_env:
        timeout_s = int(_timeout_env)
    else:
        print("Error: --timeout_s not provided and QUERY_TIMEOUT_S is not set.", file=sys.stderr)
        return 1
    args.timeout_s = timeout_s

    # Resolve concurrency: CLI arg > QUERY_CONCURRENCY env var.
    _concurrency_env = os.environ.get("QUERY_CONCURRENCY")
    if args.concurrency is not None:
        pass
    elif _concurrency_env:
        args.concurrency = int(_concurrency_env)
    else:
        print("Error: --concurrency not provided and QUERY_CONCURRENCY is not set.", file=sys.stderr)
        return 1

    # Bridge LLMOP_* → BlendSQL internal env vars so callers only need LLMOP_* names.
    if os.getenv("LLMOP_CONCURRENCY") and not os.getenv("BLENDSQL_ASYNC_LIMIT"):
        os.environ["BLENDSQL_ASYNC_LIMIT"] = os.environ["LLMOP_CONCURRENCY"]

    # Resolve db_dir: CLI arg > DB_FILES_DIR env var.
    raw_db_dir = args.db_dir or os.environ.get("DB_FILES_DIR")
    if not raw_db_dir:
        print("Error: --db_dir not provided and DB_FILES_DIR is not set.", file=sys.stderr)
        return 1
    db_dir = Path(raw_db_dir)
    if not db_dir.is_absolute():
        db_dir = (repo_root / db_dir).resolve()
    if args.db_dir is not None:
        os.environ["DB_FILES_DIR"] = str(db_dir)
    else:
        os.environ.setdefault("DB_FILES_DIR", str(db_dir))

    out_dir_raw = str(args.out_dir).strip() or _default_out_dir
    out_p = Path(out_dir_raw)
    out_dir = (repo_root / out_p).resolve() if not out_p.is_absolute() else out_p.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_config_snapshot(out_dir, script=__file__, args_dict=vars(args))

    want_qids = [str(x).strip() for x in (args.run_id or []) if str(x).strip()]
    want_dbs = [str(x).strip() for x in (args.run_db or []) if str(x).strip()]
    if want_qids and want_dbs:
        print("Note: ignoring --run_db because --run_id was provided.", file=sys.stderr)

    jobs = _read_jobs(query_file, start=0, limit=None)
    if want_qids:
        qid_set = set(want_qids)
        jobs = [j for j in jobs if j.question_id in qid_set]
        if not jobs:
            print(f"No jobs matched --run_id in {query_file}", file=sys.stderr)
            return 2
    elif want_dbs:
        db_norm_set = {_norm_db_key(x) for x in want_dbs}
        jobs = [j for j in jobs if _norm_db_key(j.db) in db_norm_set]
        if not jobs:
            print(f"No jobs matched --run_db in {query_file}", file=sys.stderr)
            return 2
    if not jobs:
        print(f"No jobs found in {query_file}", file=sys.stderr)
        return 2

    # Resolve DB paths up front so missing DBs fail fast.
    search_dirs: List[Path] = [db_dir]
    job_db_paths: Dict[int, str] = {}
    missing: List[str] = []
    for j in jobs:
        try:
            pth = _resolve_duckdb_path(j.db, repo_root=repo_root, db_dirs=search_dirs)
            job_db_paths[j.idx] = str(pth)
        except Exception as e:
            missing.append(f"{j.question_id} db={j.db}: {e}")

    if missing:
        print("Missing DuckDB files for some jobs:", file=sys.stderr)
        for line in missing[:50]:
            print(f"- {line}", file=sys.stderr)
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)", file=sys.stderr)
        print(f"Tried DB dirs: {[str(pp) for pp in search_dirs]}", file=sys.stderr)
        return 2

    # Decide per-query CSV output paths.
    used_csv_names: Dict[str, int] = {}
    job_out_paths: Dict[int, str] = {}
    for j in jobs:
        base = _csv_filename_from_question_id(j.question_id, fallback=f"line_{j.idx}")
        n = used_csv_names.get(base, 0)
        used_csv_names[base] = n + 1
        if n == 0:
            fn = base
        else:
            stem = base[:-4] if base.lower().endswith(".csv") else base
            fn = f"{stem}__{n+1}.csv"
        job_out_paths[j.idx] = str((out_dir / fn).resolve())

    # Decide per-query agent run dirs.
    agent_runs_root = (out_dir / "agent_runs").resolve()
    agent_runs_root.mkdir(parents=True, exist_ok=True)
    used_run_names: Dict[str, int] = {}
    job_run_dirs: Dict[int, str] = {}
    for j in jobs:
        ts = _timestamp_id()
        base = _run_folder_from_question_id(j.question_id, fallback=f"line_{j.idx}", ts=ts)
        n = used_run_names.get(base, 0)
        used_run_names[base] = n + 1
        folder = base if n == 0 else f"{base}__{n+1}"
        job_run_dirs[j.idx] = str((agent_runs_root / folder).resolve())

    # Clear BlendSQL disk cache so every LLM call is measured accurately.
    _clear_blendsql_cache()

    # Persist manifest for reproducibility.
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for j in jobs:
            mf.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")

    # Build extra env for subprocesses.
    extra_env: Dict[str, str] = {}
    if args.agent_model:
        extra_env["AGENT_MODEL"] = str(args.agent_model)
    if args.llmop_model:
        extra_env["LLMOP_MODEL"] = str(args.llmop_model)

    # Pre-compute per-job stdout/stderr paths (streamed live by _run_one_script).
    job_stdout_paths: Dict[int, str] = {}
    job_stderr_paths: Dict[int, str] = {}
    for j in jobs:
        stem = f"{j.idx:06d}_{_safe_slug(j.question_id)}_{_safe_slug(j.db)}"
        job_stdout_paths[j.idx] = str((out_dir / f"{stem}.stdout.txt").resolve())
        job_stderr_paths[j.idx] = str((out_dir / f"{stem}.stderr.txt").resolve())

    # Run.
    max_workers = max(1, int(args.concurrency))
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_job = {
            ex.submit(
                _run_one_script,
                job=j,
                agent_script="baseline/agentic_blendsql.py",
                db_path=str(job_db_paths.get(j.idx) or ""),
                out_dir=str(out_dir),
                output_path=str(job_out_paths.get(j.idx) or ""),
                run_dir=str(job_run_dirs.get(j.idx) or ""),
                timeout_s=int(args.timeout_s),
                stdout_path=str(job_stdout_paths.get(j.idx) or ""),
                stderr_path=str(job_stderr_paths.get(j.idx) or ""),
                extra_env=extra_env if extra_env else None,
            ): j
            for j in jobs
        }

        with results_path.open("w", encoding="utf-8") as rf, failures_path.open("w", encoding="utf-8") as ff:
            it = as_completed(fut_to_job)
            pbar = None
            if tqdm is not None:
                try:
                    pbar = tqdm(total=len(jobs), desc="BlendSQL agent queries", unit="query", dynamic_ncols=True)
                    pbar.set_postfix(ok=n_ok, fail=n_fail)
                except Exception:
                    pbar = None

            try:
                for fut in it:
                    j = fut_to_job[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {
                            "ok": False,
                            "returncode": None,
                            "elapsed_s": None,
                            "stdout": "",
                            "stderr": f"worker_error: {e}",
                            "job": {"idx": j.idx, "question_id": j.question_id, "db": j.db},
                        }

                    rf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    rf.flush()
                    if res.get("ok"):
                        n_ok += 1
                    else:
                        n_fail += 1
                        ff.write(json.dumps(res, ensure_ascii=False) + "\n")
                        ff.flush()

                    done = n_ok + n_fail
                    if pbar is not None:
                        try:
                            pbar.update(1)
                            pbar.set_postfix(ok=n_ok, fail=n_fail)
                        except Exception:
                            pass
                    else:
                        if done % 10 == 0 or done == len(jobs):
                            print(f"[{done}/{len(jobs)}] ok={n_ok} fail={n_fail}", file=sys.stderr)
            finally:
                if pbar is not None:
                    try:
                        pbar.close()
                    except Exception:
                        pass

    print(f"Done. ok={n_ok} fail={n_fail}. Outputs in: {out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
