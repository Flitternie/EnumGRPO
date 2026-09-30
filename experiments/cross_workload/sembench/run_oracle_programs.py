#!/usr/bin/env python3
"""Run the vendored SemBench Movie oracle programs with EnumGRPO's LLM endpoint.

The fixed LOTUS and Palimpzest programs live in ``oracle_programs``. Only the
model adapter and output location are overridden, so the agent/planner model is
not invoked; semantic operators use the repository's LLMOP_* configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
ORACLE_PROGRAMS = Path(__file__).resolve().parent / "oracle_programs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("lotus", "palimpzest"), required=True)
    parser.add_argument(
        "--sembench-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "sembench" / "source",
    )
    parser.add_argument("--scale-factor", type=int, default=2000)
    parser.add_argument("--queries", default="1", help="Comma-separated query IDs")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")


def load_oracle_class(filename: str, class_name: str) -> type:
    """Load a vendored oracle program without shadowing SemBench packages."""
    path = ORACLE_PROGRAMS / filename
    spec = importlib.util.spec_from_file_location(f"sembench_oracle_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load oracle program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    return float(value) if value else default


def reprice(usage: dict[str, int]) -> float:
    prompt = int(usage.get("input_tokens", 0) or 0)
    cached = int(usage.get("cached_tokens", 0) or 0)
    uncached = max(0, prompt - cached)
    output = int(usage.get("output_tokens", 0) or 0)
    return (
        uncached * env_float("LLMOP_USD_PER_1M_INPUT", 1.0) / 1_000_000
        + cached * env_float("LLMOP_USD_PER_1M_CACHE_READ", 0.1) / 1_000_000
        + output * env_float("LLMOP_USD_PER_1M_OUTPUT", 5.0) / 1_000_000
    )


def run_lotus(args: argparse.Namespace, query_ids: list[int]) -> list[dict[str, Any]]:
    system_root = Path(
        os.getenv("LOTUS_SYSTEM_DIR", REPO_ROOT / "baseline" / "lotus" / "system")
    ).expanduser().resolve()
    sys.path[:0] = [str(system_root), str(args.sembench_root / "src")]

    from lotus.models import LM

    LotusRunner = load_oracle_class("lotus.py", "LotusRunner")

    class AlignedLotusRunner(LotusRunner):
        def _configure_lm(self):
            model = (os.getenv("LLMOP_MODEL") or "").strip()
            api_key = (os.getenv("LLMOP_API_KEY") or "").strip()
            base_url = (os.getenv("LLMOP_BASE_URL") or "").strip()
            if not model or not api_key or not base_url:
                raise RuntimeError("LLMOP_MODEL, LLMOP_API_KEY, and LLMOP_BASE_URL are required")

            class CountingLM(LM):
                def __init__(self, *lm_args, **lm_kwargs):
                    super().__init__(*lm_args, **lm_kwargs)
                    self.logical_calls = 0

                def __call__(self, messages, *call_args, **call_kwargs):
                    self.logical_calls += len(messages)
                    return super().__call__(messages, *call_args, **call_kwargs)

            return CountingLM(
                model=model,
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                max_batch_size=self.concurrent_llm_worker,
                max_tokens=self.max_tokens,
            )

        def _update_token_usage(self, metric):
            physical = self.lm.stats.physical_usage
            usage = {
                "input_tokens": int(getattr(physical, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(physical, "completion_tokens", 0) or 0),
                "cached_tokens": int(getattr(physical, "cached_prompt_tokens", 0) or 0),
                "llm_calls": int(getattr(self.lm, "logical_calls", 0) or 0),
            }
            metric.token_usage = usage["input_tokens"] + usage["output_tokens"]
            metric.money_cost = reprice(usage)
            self.last_usage = usage

    runner = AlignedLotusRunner(
        "movie",
        args.scale_factor,
        model_name=(os.getenv("LLMOP_MODEL") or "").strip(),
        concurrent_llm_worker=int(os.getenv("LLMOP_CONCURRENCY") or "32"),
        skip_setup=True,
    )
    records: list[dict[str, Any]] = []
    for query_id in query_ids:
        runner.lm.reset_cache()
        runner.lm.logical_calls = 0
        runner.last_usage = {}
        metric = runner.execute_query(query_id)
        usage = dict(runner.last_usage)
        frame = metric.results
        frame.to_csv(args.out_dir / f"Q{query_id}.csv", index=False)
        records.append(
            {
                **metric.to_dict(),
                "system": "lotus",
                "model": (os.getenv("LLMOP_MODEL") or "").strip(),
                "usage": usage,
                "cost_usd_repriced": reprice(usage),
            }
        )
        runner.lm.reset_cache()
    return records


def run_palimpzest(
    args: argparse.Namespace, query_ids: list[int]
) -> list[dict[str, Any]]:
    baseline_root = REPO_ROOT / "baseline" / "palimpzest"
    system_checkout = Path(
        os.getenv("PALIMPZEST_SYSTEM_DIR", baseline_root / "system")
    ).expanduser().resolve()
    system_root = system_checkout / "src"
    sys.path[:0] = [str(baseline_root), str(system_root), str(args.sembench_root / "src")]

    import common as pz_common

    pz_common.install_pz_integration()
    PalimpzestRunner = load_oracle_class("palimpzest.py", "PalimpzestRunner")

    class AlignedPalimpzestRunner(PalimpzestRunner):
        def palimpzest_config(self):
            return pz_common.build_config()

    runner = AlignedPalimpzestRunner(
        "movie",
        args.scale_factor,
        model_name=(os.getenv("LLMOP_MODEL") or "").strip(),
        concurrent_llm_worker=int(os.getenv("LLMOP_CONCURRENCY") or "32"),
        skip_setup=True,
    )
    records: list[dict[str, Any]] = []
    for query_id in query_ids:
        pz_common.reset_usage()
        metric = runner.execute_query(query_id)
        raw = pz_common.snapshot_usage()
        usage = {
            "input_tokens": int(raw.get("input_text_tokens", 0) or 0),
            "output_tokens": int(raw.get("output_text_tokens", 0) or 0),
            "cached_tokens": int(raw.get("cache_read_tokens", 0) or 0),
            "cache_creation_tokens": int(raw.get("cache_creation_tokens", 0) or 0),
            "llm_calls": int(raw.get("total_llm_calls", 0) or 0),
        }
        metric.token_usage = usage["input_tokens"] + usage["output_tokens"]
        metric.money_cost = reprice(usage)
        metric.results.to_csv(args.out_dir / f"Q{query_id}.csv", index=False)
        records.append(
            {
                **metric.to_dict(),
                "system": "palimpzest",
                "model": (os.getenv("LLMOP_MODEL") or "").strip(),
                "usage": usage,
                "cost_usd_repriced": reprice(usage),
            }
        )
    return records


def main() -> int:
    args = parse_args()
    load_environment()
    args.sembench_root = args.sembench_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    query_values = args.queries.replace(":", ",")
    query_ids = [int(value.strip()) for value in query_values.split(",") if value.strip()]
    if not query_ids or any(value < 1 or value > 10 for value in query_ids):
        raise ValueError("--queries must contain IDs from 1 through 10")

    records = (
        run_lotus(args, query_ids)
        if args.system == "lotus"
        else run_palimpzest(args, query_ids)
    )
    output = args.out_dir / "metrics.json"
    merged: dict[int, dict[str, Any]] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        merged.update({int(record["query_id"]): record for record in previous})
    merged.update({int(record["query_id"]): record for record in records})
    final_records = [merged[query_id] for query_id in sorted(merged)]
    output.write_text(
        json.dumps(final_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
