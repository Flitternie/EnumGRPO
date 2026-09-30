#!/usr/bin/env python3
"""Aggregate Spider pure-SQL runs 1..5 without replacing the K=3 report."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from aggregate_runs import (  # noqa: E402
    _aggregate,
    _extract_metrics,
    _load_summary,
    _print_table,
)


def main() -> int:
    base_dir = REPO_ROOT / "exp/eval/pure_sql_spider"
    agents = ["agent_no_pool", "agent_with_swan_pool"]
    k = 5
    aggregate = {}
    n_total = None

    for slug in agents:
        metrics = []
        for run_number in range(1, k + 1):
            summary_path = base_dir / slug / f"run_{run_number}" / "eval_summary.json"
            summary = _load_summary(summary_path)
            if summary is None:
                raise FileNotFoundError(f"missing or invalid summary: {summary_path}")
            if n_total is None:
                n_total = summary.get("n_total")
            elif summary.get("n_total") != n_total:
                raise ValueError(
                    f"n_total mismatch in {summary_path}: "
                    f"{summary.get('n_total')} != {n_total}"
                )
            metrics.append(_extract_metrics(summary))
        aggregate[slug] = _aggregate(metrics)

    _print_table(agents, aggregate, k=k, n_total=n_total)

    def clean(value: float):
        return None if math.isnan(value) else value

    payload = {
        "base_dir": str(base_dir),
        "k": k,
        "agents": agents,
        "n_total": n_total,
        "aggregate": {
            slug: {
                metric: {stat: clean(float(value)) for stat, value in stats.items()}
                for metric, stats in agent_metrics.items()
            }
            for slug, agent_metrics in aggregate.items()
        },
    }
    output_path = base_dir / "aggregate_summary_k5.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Aggregate summary saved to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
