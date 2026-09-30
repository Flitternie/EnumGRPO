#!/usr/bin/env python3
"""Aggregate repeated SemBench Movie runs by condition."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = (
    "agentic_text2sql",
    "agent_no_pool",
    "agent_with_swan_experiences",
)

METRICS = {
    "retrieval_f1": ("official_metric_means", "retrieval_f1"),
    "aggregation_relative_error": ("official_metric_means", "aggregation_relative_error"),
    "ranking_spearman": ("official_metric_means", "ranking_spearman"),
    "ranking_kendall_tau": ("official_metric_means", "ranking_kendall_tau"),
    "mean_elapsed_s": ("runtime", "mean_elapsed_s"),
    "total_planner_cost_usd": ("runtime", "total_planner_cost_usd"),
    "total_llm_op_cost_usd": ("runtime", "total_llm_op_cost_usd"),
    "total_llm_op_calls": ("runtime", "total_llm_op_calls"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _quality(payload: dict[str, Any]) -> float:
    """Compute SemBench's per-query quality average."""
    values: list[float] = []
    for query in payload["queries"]:
        metrics = query["metrics"]
        if "f1_score" in metrics:
            value = float(metrics["f1_score"])
        elif "relative_error" in metrics:
            value = 1.0 / (1.0 + float(metrics["relative_error"]))
        else:
            value = max(0.0, float(metrics["spearman_correlation"]))
        values.append(value)
    return statistics.fmean(values)


def main() -> int:
    args = _args()
    root = args.experiment_root.resolve()
    result: dict[str, Any] = {
        "benchmark": "SemBench",
        "scenario": "movie",
        "runs": args.runs,
        "conditions": {},
    }
    for condition in CONDITIONS:
        summaries: list[dict[str, Any]] = []
        paths: list[str] = []
        for run_number in range(1, args.runs + 1):
            path = root / condition / f"run_{run_number}" / "eval_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("n_queries", 0)) != 10:
                raise ValueError(f"Expected 10 evaluated queries in {path}")
            summaries.append(payload)
            paths.append(str(path))

        aggregate: dict[str, dict[str, float | int]] = {
            "quality": _stats([_quality(payload) for payload in summaries]),
        }
        for name, (section, field) in METRICS.items():
            values = [float(payload[section][field]) for payload in summaries]
            aggregate[name] = _stats(values)
        result["conditions"][condition] = {
            "run_summaries": paths,
            "aggregate": aggregate,
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
