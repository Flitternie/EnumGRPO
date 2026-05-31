#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from utils.prompt import resolve_prompt
from utils.snapshot import save_config_snapshot as _save_config_snapshot


@dataclass(frozen=True)
class Job:
    idx: int
    question_id: str
    db: str
    query: str
    hint: str
    required_columns: Optional[List[str]] = None


def _repo_root() -> Path:
    # This script lives at the repo root.
    return Path(__file__).resolve().parent


def _agent_dir(repo_root: Path) -> Path:
    return (repo_root / "agent").resolve()


def _read_jobs(path: Path, *, start: int = 0, limit: Optional[int] = None) -> List[Job]:
    jobs: List[Job] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit is not None and len(jobs) >= limit:
                break
            s = (line or "").strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                continue
            qid = str(obj.get("question_id") or f"line_{i}").strip()
            db = str(obj.get("db") or "").strip()
            query = str(obj.get("query") or "").strip()
            hint = str(obj.get("hint") or "").strip()
            required_columns = obj.get("required_columns") or None
            if not db or not query:
                continue
            jobs.append(Job(
                idx=i, question_id=qid, db=db,
                query=query, hint=hint,
                required_columns=required_columns,
            ))
    return jobs


def _safe_slug(s: str) -> str:
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_.")
    return slug or "item"


def _csv_filename_from_question_id(question_id: str, *, fallback: str) -> str:
    """
    Create a per-query CSV filename derived from question_id.

    We keep question_id mostly intact, but replace path separators and control chars
    so we never escape the output directory.
    """
    s = (question_id or "").strip() or fallback
    cleaned: List[str] = []
    for ch in s:
        if ch in {"/", "\\", "\x00"} or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    name = "".join(cleaned).strip().strip(".")
    if not name:
        name = fallback
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


def _run_folder_from_question_id(question_id: str, *, fallback: str, ts: str) -> str:
    """
    Create a per-query run folder name derived from question_id and timestamp.
    """
    s = (question_id or "").strip() or fallback
    cleaned: List[str] = []
    for ch in s:
        if ch in {"/", "\\", "\x00"} or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    base = "".join(cleaned).strip().strip(".") or fallback
    return f"{base}_{ts}"


def _timestamp_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _build_message(job: Job) -> str:
    parts = [
        f"Question ID: {job.question_id}",
        f"DB: {job.db}",
        "",
        "Task:",
        job.query,
    ]
    if job.hint:
        parts.extend(["", f"Hint: {job.hint}"])
    if job.required_columns:
        cols = ", ".join(job.required_columns)
        parts.extend(["", f"Required output columns: {cols}"])
    parts.extend(
        [
            "",
            "Return the final answer clearly. If the result is a table, show the table rows.",
        ]
    )
    return "\n".join(parts).strip()


def _norm_db_key(s: str) -> str:
    """
    Normalize a DB id / filename stem for fuzzy matching.
    """
    out: List[str] = []
    for ch in (s or ""):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    return "".join(out).strip("_")


def _resolve_duckdb_path(name_or_path: str, *, repo_root: Path, db_dirs: List[Path]) -> Path:
    """
    Resolve a DuckDB file path based on a db id (like 'california_schools') by
    scanning one or more directories for a matching `.duckdb` file.

    Matching rules:
    - exact file path if `name_or_path` looks like a path and exists
    - case-insensitive match on stem (with or without `.duckdb`)
    - normalized match (non-alnum treated as `_`)
    """
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ValueError("db name is empty")

    # If user provided a direct path, respect it (absolute, or relative to repo root).
    if ("/" in raw) or ("\\" in raw) or raw.lower().endswith(".duckdb"):
        p = Path(raw)
        cand = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
        if cand.is_file():
            return cand

    want_stem = raw[:-7] if raw.lower().endswith(".duckdb") else raw
    want_upper = want_stem.upper()
    want_norm = _norm_db_key(want_stem)

    tried_dirs: List[str] = []
    for d in db_dirs:
        d2 = d.resolve()
        tried_dirs.append(str(d2))
        if not d2.exists() or not d2.is_dir():
            continue
        try:
            for p in d2.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() != ".duckdb":
                    continue
                stem = p.stem
                if stem.upper() == want_upper:
                    return p.resolve()
                if _norm_db_key(stem) == want_norm:
                    return p.resolve()
        except Exception:
            continue

    # Not found.
    raise FileNotFoundError(f"duckdb not found for db='{raw}' (tried: {', '.join(tried_dirs)})")


def _stream_to_file(stream, dest_file, buf: list) -> None:
    """Read lines from *stream* and write them to *dest_file* and *buf* in real-time."""
    try:
        for line in stream:
            dest_file.write(line)
            dest_file.flush()
            buf.append(line)
    except Exception:
        pass


def _run_agent(
    *,
    job: "Job",
    cmd: List[str],
    cwd: str,
    out_dir: str,
    db_path_s: str,
    output_path: str,
    run_dir: str,
    timeout_s: int,
    stdout_path: str,
    stderr_path: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Core subprocess runner: Popen + streaming threads + explicit kill on timeout.

    Callers are responsible for building *cmd* and choosing *cwd*; this function
    handles everything from launch through the return dict.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if extra_env:
        env.update({str(k): str(v) for (k, v) in extra_env.items()})

    t0 = time.time()
    stdout_buf: list = []
    stderr_buf: list = []
    timed_out = False

    with (
        open(stdout_path, "w", encoding="utf-8", buffering=1) as sout_f,
        open(stderr_path, "w", encoding="utf-8", buffering=1) as serr_f,
    ):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            t_out = threading.Thread(target=_stream_to_file, args=(proc.stdout, sout_f, stdout_buf), daemon=True)
            t_err = threading.Thread(target=_stream_to_file, args=(proc.stderr, serr_f, stderr_buf), daemon=True)
            t_out.start()
            t_err.start()
            try:
                proc.wait(timeout=int(timeout_s))
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()
            finally:
                t_out.join(timeout=5)
                t_err.join(timeout=5)
        except Exception as exc:
            elapsed_s = time.time() - t0
            return {
                "ok": False,
                "returncode": None,
                "elapsed_s": float(elapsed_s),
                "stdout": "".join(stdout_buf),
                "stderr": "".join(stderr_buf) + f"\nworker_error: {exc}",
                "job": {
                    "idx": job.idx,
                    "question_id": job.question_id,
                    "db": job.db,
                    "db_path": db_path_s,
                    "output_path": str(output_path),
                    "run_dir": str(run_dir),
                },
            }

    elapsed_s = time.time() - t0
    stdout_text = "".join(stdout_buf)
    stderr_text = "".join(stderr_buf)
    if timed_out:
        stderr_text += "\nTIMEOUT"
        with open(stderr_path, "a", encoding="utf-8") as serr_f:
            serr_f.write("\nTIMEOUT\n")
    return {
        "ok": (not timed_out) and proc.returncode == 0,
        "returncode": None if timed_out else int(proc.returncode),
        "elapsed_s": float(elapsed_s),
        "stdout": stdout_text,
        "stderr": stderr_text,
        "job": {
            "idx": job.idx,
            "question_id": job.question_id,
            "db": job.db,
            "db_path": db_path_s,
            "output_path": str(output_path),
            "run_dir": str(run_dir),
        },
    }


def _run_one_script(
    *,
    job: "Job",
    agent_script: str,
    db_path: str,
    out_dir: str,
    output_path: str,
    run_dir: str,
    timeout_s: int,
    stdout_path: str,
    stderr_path: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Runner for script-based agents (agentic_text2sql, agentic_blendsql).

    Invokes: python <repo_root>/<agent_script> db --db_path ... --message ... --auto
    """
    repo_root = _repo_root()
    db_path_s = str(db_path or "").strip()
    if not db_path_s:
        raise ValueError("db_path is empty")
    msg = _build_message(job)
    cmd = [
        sys.executable,
        str((repo_root / agent_script).resolve()),
        "db",
        "--db_path", db_path_s,
        "--message", msg,
        "--output_path", str(output_path),
    ]
    if str(run_dir or "").strip():
        cmd.extend(["--run_dir", str(run_dir)])
    cmd.extend(["--auto"])
    return _run_agent(
        job=job, cmd=cmd, cwd=str(repo_root),
        out_dir=out_dir, db_path_s=db_path_s, output_path=output_path, run_dir=run_dir,
        timeout_s=timeout_s, stdout_path=stdout_path, stderr_path=stderr_path,
        extra_env=extra_env,
    )


def _run_one(
    *,
    job: Job,
    db_path: str,
    out_dir: str,
    output_path: str,
    run_dir: str,
    timeout_s: int,
    stdout_path: str,
    stderr_path: str,
    system_prompt_path: str = "",
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Runner for the DB agent invoked as a module (python -m codebase db ...)."""
    repo_root = _repo_root()
    agent_dir = _agent_dir(repo_root)
    db_path_s = str(db_path or "").strip()
    if not db_path_s:
        raise ValueError("db_path is empty")
    msg = _build_message(job)
    cmd = [
        sys.executable,
        "-m", "codebase.main_noschemtools",
        "db",
        "--db_path", db_path_s,
        "--message", msg,
        "--output_path", str(output_path),
    ]
    if str(run_dir or "").strip():
        cmd.extend(["--run_dir", str(run_dir)])
    if str(system_prompt_path or "").strip():
        cmd.extend(["--system_prompt", str(Path(system_prompt_path).resolve())])
    cmd.extend(["--auto"])
    return _run_agent(
        job=job, cmd=cmd, cwd=str(agent_dir),
        out_dir=out_dir, db_path_s=db_path_s, output_path=output_path, run_dir=run_dir,
        timeout_s=timeout_s, stdout_path=stdout_path, stderr_path=stderr_path,
        extra_env=extra_env,
    )


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = _repo_root()
    agent_dir = _agent_dir(repo_root)
    if not agent_dir.is_dir():
        print(f"Expected agent directory at {agent_dir} but it does not exist.", file=sys.stderr)
        return 2
    p = argparse.ArgumentParser(description="Run agent on swan/swan.jsonl with multiprocessing.")
    p.add_argument("--query_file", default=str((repo_root / "swan" / "swan.jsonl").resolve()),
                   help="Path to the input JSONL file (default: swan/swan.jsonl).")
    p.add_argument(
        "--db_dir",
        default=None,
        help="Directory containing .duckdb files. Falls back to DB_FILES_DIR env var.",
    )
    _default_out_dir = str((repo_root / "swan" / "logs" / f"run_{_timestamp_id()}").resolve())
    p.add_argument(
        "--out_dir",
        default=_default_out_dir,
        help="Output root dir for SWAN artifacts + per-query agent run logs (default: swan/logs/run_<timestamp>).",
    )
    p.add_argument(
        "--agent_model",
        default=None,
        help="Sets AGENT_MODEL for the agent subprocesses (optional if already in .env or environment).",
    )
    p.add_argument(
        "--prompt_file",
        default=None,
        help=(
            "System prompt source. Three formats are accepted: "
            "(1) a .md file -- used as-is for every query; "
            "(2) a .json file with an 'experiences' key -- global experiences appended to the base prompt; "
            "(3) a .json file with a 'db_experiences' key -- per-database experiences, "
            "each query receives only the experiences for its own database."
        ),
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
        return 2
    args.timeout_s = timeout_s

    # Resolve concurrency: CLI arg > QUERY_CONCURRENCY env var.
    _concurrency_env = os.environ.get("QUERY_CONCURRENCY")
    if args.concurrency is not None:
        pass
    elif _concurrency_env:
        args.concurrency = int(_concurrency_env)
    else:
        print("Error: --concurrency not provided and QUERY_CONCURRENCY is not set.", file=sys.stderr)
        return 2

    # Resolve db_dir: CLI arg > DB_FILES_DIR env var.
    raw_db_dir = args.db_dir or os.environ.get("DB_FILES_DIR")
    if not raw_db_dir:
        print("Error: --db_dir not provided and DB_FILES_DIR is not set.", file=sys.stderr)
        return 2
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

    # --- Resolve the system prompt (auto-detects .md / .json / directory) ---
    resolved_prompt = ""
    db_prompt_map: Dict[str, str] = {}
    if args.prompt_file:
        try:
            resolved_prompt, db_prompt_map = resolve_prompt(args.prompt_file, out_dir)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: --prompt_file: {exc}", file=sys.stderr)
            return 2
        if db_prompt_map:
            print(
                f"DB-specific experiences loaded for {len(db_prompt_map)} database(s): "
                f"{', '.join(sorted(db_prompt_map))}",
                file=sys.stderr,
            )
        elif resolved_prompt:
            print(f"Prompt resolved: {resolved_prompt}", file=sys.stderr)

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

    # Resolve DB paths up front so missing DBs fail fast (before launching workers).
    # We only search in --db_dir (or DB_FILES_DIR).
    search_dirs: List[Path] = [db_dir]
    job_db_paths: Dict[int, str] = {}
    missing: List[str] = []
    for j in jobs:
        try:
            p = _resolve_duckdb_path(j.db, repo_root=repo_root, db_dirs=search_dirs)
            job_db_paths[j.idx] = str(p)
        except Exception as e:
            missing.append(f"{j.question_id} db={j.db}: {e}")

    if missing:
        print("Missing DuckDB files for some jobs:", file=sys.stderr)
        for line in missing[:50]:
            print(f"- {line}", file=sys.stderr)
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)", file=sys.stderr)
        print(f"Tried DB dirs: {[str(p) for p in search_dirs]}", file=sys.stderr)
        print("Tip: pass --db_dir or set DB_FILES_DIR to the directory containing your SWAN .duckdb files.", file=sys.stderr)
        return 2

    # Decide per-query CSV output paths up front to avoid collisions under multiprocessing.
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

    # Decide per-query agent run dirs (always under out_dir).
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

    # Persist manifest for reproducibility.
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for j in jobs:
            mf.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")

    # In DB-specific mode, verify every job's db has a matching experience entry.
    if db_prompt_map:
        unmatched = sorted({
            j.db for j in jobs
            if not (db_prompt_map.get(_norm_db_key(j.db)) or db_prompt_map.get(j.db))
        })
        if unmatched:
            print(
                f"Error: the following databases in '{args.query_file}' have no matching "
                f"entry in the experience file:\n"
                + "\n".join(f"  - {db}" for db in unmatched)
                + f"\nAvailable keys: {', '.join(sorted(db_prompt_map))}",
                file=sys.stderr,
            )
            return 2

    # Run.
    max_workers = max(1, int(args.concurrency))
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_job = {
            ex.submit(
                _run_one,
                job=j,
                db_path=str(job_db_paths.get(j.idx) or ""),
                out_dir=str(out_dir),
                output_path=str(job_out_paths.get(j.idx) or ""),
                run_dir=str(job_run_dirs.get(j.idx) or ""),
                timeout_s=int(args.timeout_s),
                stdout_path=str(out_dir / f"{j.idx:06d}_{_safe_slug(j.question_id)}_{_safe_slug(j.db)}.stdout.txt"),
                stderr_path=str(out_dir / f"{j.idx:06d}_{_safe_slug(j.question_id)}_{_safe_slug(j.db)}.stderr.txt"),
                system_prompt_path=(
                    db_prompt_map.get(_norm_db_key(j.db))
                    or db_prompt_map.get(j.db)
                ) if db_prompt_map else resolved_prompt,
                extra_env=({"AGENT_MODEL": str(args.agent_model)} if args.agent_model else None),
            ): j
            for j in jobs
        }

        with results_path.open("w", encoding="utf-8") as rf, failures_path.open("w", encoding="utf-8") as ff:
            it = as_completed(fut_to_job)
            pbar = None
            if tqdm is not None:
                try:
                    pbar = tqdm(total=len(jobs), desc="SWAN queries", unit="query", dynamic_ncols=True)
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

