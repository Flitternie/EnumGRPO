#!/usr/bin/env python3
"""Aggregate axis-ablation runs and optionally apply GPT pricing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


CONDITIONS = [
    "without_execution_paradigm",
    "without_operator_type",
    "without_operator_placement",
    "without_selectivity_scope",
    "without_projection_width",
]


def _fmt(metric: dict[str, float], digits: int = 2, scale: float = 1.0) -> str:
    return f"{metric['mean'] * scale:.{digits}f} +/- {metric['sd'] * scale:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpt-pricing", action="store_true")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    base = args.base_dir.resolve()
    conditions = args.condition or CONDITIONS
    aggregate_cmd = [
        sys.executable,
        str(repo / "aggregate_runs.py"),
        "--base_dir",
        str(base),
        "--agents",
        *conditions,
        "--k",
        str(args.runs),
    ]
    subprocess.run(aggregate_cmd, cwd=repo, check=True, stdout=subprocess.DEVNULL)

    if args.gpt_pricing:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            pricing_cmd = [
                sys.executable,
                str(repo / "experiments/gpt_transfer/reprice_gpt_costs.py"),
                "--experiment-root",
                str(base),
                "--runs",
                str(args.runs),
                "--planner-model",
                "gpt-5.6-sol",
                "--operator-model",
                "gpt-5.6-luna",
                "--update-aggregate-summaries",
                "--output",
                tmp.name,
            ]
            for condition in conditions:
                pricing_cmd.extend(["--condition", f"{condition}={condition}"])
            subprocess.run(pricing_cmd, cwd=repo, check=True, stdout=subprocess.DEVNULL)

    payload = json.loads((base / "aggregate_summary.json").read_text(encoding="utf-8"))
    n_total = int(payload["n_total"])
    lines = [
        "# Experience-axis ablation",
        "",
        f"Mean +/- sample SD over {args.runs} runs. LLM-op cost is the total over {n_total} queries.",
        "",
        "| Condition | EX (%) | Tuple-F1 (%) | Cell-F1 (%) | Walltime (s) | LLM-op cost ($) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        metrics = payload["aggregate"][condition]
        lines.append(
            "| " + condition + " | "
            + " | ".join(
                [
                    _fmt(metrics["success_rate"]),
                    _fmt(metrics["row_level_f1"]),
                    _fmt(metrics["item_level_f1"]),
                    _fmt(metrics["avg_elapsed_s"]),
                    _fmt(metrics["llmop_cost_amortised"], digits=4, scale=n_total),
                ]
            )
            + " |"
        )
    (base / "aggregate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
