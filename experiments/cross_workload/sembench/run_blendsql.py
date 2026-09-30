#!/usr/bin/env python3
"""Run the published BlendSQL programs on the SemBench Movie workload."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import duckdb
from blendsql import BlendSQL, config
from blendsql.db import DuckDB
from blendsql.models import VLLM
from dotenv import load_dotenv


SOURCE_REPOSITORY = "https://github.com/CapitalOne-Research/play-by-the-type-rules"
SOURCE_REVISION = "6e1b3606e37ac378131e908b40ac290c1440a7d7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--queries-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def env_float(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file, override=False)

    required = ("LLMOP_MODEL", "LLMOP_BASE_URL", "LLMOP_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    db_path = args.db.resolve()
    queries_dir = args.queries_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    query_paths = [queries_dir / f"Q{index}.sql" for index in range(1, 11)]
    missing_queries = [str(path) for path in query_paths if not path.is_file()]
    if missing_queries:
        raise FileNotFoundError(f"Missing query programs: {missing_queries}")

    model_name = os.environ["LLMOP_MODEL"].removeprefix("openai/")
    input_price = env_float("LLMOP_USD_PER_1M_INPUT")
    cache_read_price = env_float("LLMOP_USD_PER_1M_CACHE_READ")
    output_price = env_float("LLMOP_USD_PER_1M_OUTPUT")

    write_json(
        out_dir / "config_snapshot.json",
        {
            "system": "BlendSQL",
            "framework_version": "0.1.26",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "database": str(db_path),
            "queries_dir": str(queries_dir),
            "query_count": len(query_paths),
            "llm_operator_model": model_name,
            "concurrency": args.concurrency,
            "enable_cascade_filter": True,
            "enable_early_exit": True,
            "enable_constrained_decoding": False,
            "pricing_per_million_tokens": {
                "input": input_price,
                "cache_read": cache_read_price,
                "output": output_price,
            },
        },
    )

    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        metrics: list[dict[str, Any]] = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = []
    completed = {
        int(record["query_id"])
        for record in metrics
        if record.get("status") == "success"
    }

    config.set_deterministic(True)
    config.set_async_limit(args.concurrency)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        connection.execute("SELECT setseed(0.5)")
        model = VLLM(
            model_name_or_path=model_name,
            base_url=os.environ["LLMOP_BASE_URL"],
            api_key=os.environ["LLMOP_API_KEY"],
            caching=False,
        )
        engine = BlendSQL(
            DuckDB(connection),
            model=model,
            verbose=False,
            enable_constrained_decoding=False,
            enable_cascade_filter=True,
            enable_early_exit=True,
        )

        for query_id, query_path in enumerate(query_paths, start=1):
            if query_id in completed:
                print(f"Q{query_id}: already complete; skipping", flush=True)
                continue

            before = {
                "input_tokens": model.prompt_tokens,
                "output_tokens": model.completion_tokens,
                "cached_tokens": model.cached_tokens,
                "llm_calls": model.num_generation_calls,
            }
            started = time.perf_counter()
            status = "success"
            error = None
            row_count = 0
            usage = None
            try:
                result = engine.execute(query_path.read_text(encoding="utf-8"))
                frame = result.df()
                row_count = len(frame)
                frame.to_csv(out_dir / f"Q{query_id}.csv", index=False)
                usage = {
                    "input_tokens": result.meta.prompt_tokens,
                    "output_tokens": result.meta.completion_tokens,
                    "cached_tokens": result.meta.cached_tokens,
                    "llm_calls": result.meta.num_generation_calls,
                }
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                (out_dir / f"Q{query_id}.stderr.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            elapsed = time.perf_counter() - started
            if usage is None:
                usage = {
                    "input_tokens": model.prompt_tokens - before["input_tokens"],
                    "output_tokens": model.completion_tokens - before["output_tokens"],
                    "cached_tokens": model.cached_tokens - before["cached_tokens"],
                    "llm_calls": model.num_generation_calls - before["llm_calls"],
                }
            noncached_input = max(0, usage["input_tokens"] - usage["cached_tokens"])
            cost = (
                noncached_input * input_price
                + usage["cached_tokens"] * cache_read_price
                + usage["output_tokens"] * output_price
            ) / 1_000_000
            record = {
                "query_id": query_id,
                "status": status,
                "execution_time": elapsed,
                "row_count": row_count,
                "system": "blendsql",
                "model": model_name,
                "usage": usage,
                "cost_usd_repriced": cost,
                "error": error,
            }
            metrics = [item for item in metrics if int(item["query_id"]) != query_id]
            metrics.append(record)
            metrics.sort(key=lambda item: int(item["query_id"]))
            write_json(metrics_path, metrics)
            print(
                f"Q{query_id}: {status}, rows={row_count}, elapsed={elapsed:.2f}s, "
                f"calls={usage['llm_calls']}, cost=${cost:.6f}",
                flush=True,
            )

    failures = [record for record in metrics if record["status"] != "success"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
