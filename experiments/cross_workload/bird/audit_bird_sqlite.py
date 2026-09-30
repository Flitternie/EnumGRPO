#!/usr/bin/env python3
"""Audit BIRD gold SQL against the released SQLite databases.

The audit opens every database read-only, enforces a per-query execution
timeout through SQLite's progress handler, and records enough timing and error
information to choose a reproducible evaluation subset.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


_DATABASE_ROOT: Path
_TIMEOUT_SECONDS: float


def _init_worker(database_root: str, timeout_seconds: float) -> None:
    global _DATABASE_ROOT, _TIMEOUT_SECONDS
    _DATABASE_ROOT = Path(database_root)
    _TIMEOUT_SECONDS = timeout_seconds


def _run_query(record: dict[str, Any]) -> dict[str, Any]:
    db_id = str(record["db_id"])
    db_path = _DATABASE_ROOT / db_id / f"{db_id}_template.sqlite"
    started = time.monotonic()
    row_count = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=5,
        )
        connection.set_progress_handler(
            lambda: int(time.monotonic() - started > _TIMEOUT_SECONDS),
            10_000,
        )
        cursor = connection.execute(str(record["SQL"]))
        while True:
            batch = cursor.fetchmany(10_000)
            if not batch:
                break
            row_count += len(batch)
        return {
            "question_id": record["question_id"],
            "db_id": db_id,
            "difficulty": record["difficulty"],
            "ok": True,
            "row_count": row_count,
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "question_id": record["question_id"],
            "db_id": db_id,
            "difficulty": record["difficulty"],
            "ok": False,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        if connection is not None:
            connection.close()


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_seconds"]) for row in results]
    failures = [row for row in results if not row["ok"]]
    return {
        "n_queries": len(results),
        "n_success": len(results) - len(failures),
        "n_failure": len(failures),
        "success_rate": (len(results) - len(failures)) / len(results),
        "databases": dict(sorted(Counter(row["db_id"] for row in results).items())),
        "difficulties": dict(
            sorted(Counter(row["difficulty"] for row in results).items())
        ),
        "elapsed_seconds": {
            "sum": sum(elapsed),
            "mean": statistics.mean(elapsed),
            "median": statistics.median(elapsed),
            "max": max(elapsed),
        },
        "failure_messages": dict(
            sorted(Counter(row.get("error", "") for row in failures).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument(
        "--difficulty",
        choices=("all", "simple", "moderate", "challenging"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if args.difficulty != "all":
        records = [row for row in records if row["difficulty"] == args.difficulty]

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.database_root.resolve()), args.timeout_seconds),
    ) as pool:
        results = list(pool.imap_unordered(_run_query, records))

    results.sort(key=lambda row: int(row["question_id"]))
    payload = {
        "annotations": str(args.annotations.resolve()),
        "database_root": str(args.database_root.resolve()),
        "difficulty_filter": args.difficulty,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "summary": _summary(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["summary"]["n_failure"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
