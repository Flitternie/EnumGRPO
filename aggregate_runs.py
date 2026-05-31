#!/usr/bin/env python3
"""Aggregate K repeated SWAN evaluation runs and print a comparison table.

Expected directory layout (produced by run_multi_eval.sh):

    <base_dir>/
        <agent_slug>/
            run_1/eval_summary.json
            run_2/eval_summary.json
            ...

Usage:
    python aggregate_runs.py --base_dir exp/multi_eval_... \
        --agents text2sql blendsql db_agent db_agent_lx --k 3

Outputs:
  - A markdown table (stdout) matching results_comparison.md style.
  - <base_dir>/aggregate_summary.json with raw aggregate data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 for n <= 1."""
    n = len(values)
    if n <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def _ci95(values: Sequence[float]) -> float:
    """95 percent CI half-width (t-distribution, two-tailed)."""
    n = len(values)
    if n <= 1:
        return 0.0
    T_TABLE = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000,
    }
    df = n - 1
    if df in T_TABLE:
        t = T_TABLE[df]
    elif df > 60:
        t = 1.96
    else:
        keys = sorted(T_TABLE)
        lo = max(k for k in keys if k <= df)
        hi = min(k for k in keys if k >= df)
        frac = (df - lo) / (hi - lo) if hi != lo else 0.0
        t = T_TABLE[lo] + frac * (T_TABLE[hi] - T_TABLE[lo])
    return t * _std(values) / math.sqrt(n)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric extraction from one eval_summary.json
# ---------------------------------------------------------------------------

_LLMOP_IN_PRICE          = 1.00 / 1_000_000
_LLMOP_CACHE_WRITE_PRICE = 1.25 / 1_000_000
_LLMOP_CACHE_READ_PRICE  = 0.10 / 1_000_000
_LLMOP_OUT_PRICE         = 5.00 / 1_000_000


def _extract_metrics(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    perf  = summary.get("performance") or {}
    costs = summary.get("costs") or {}
    ap    = costs.get("agent_planning") or {}
    lo    = costs.get("llm_op") or {}

    n_total  = summary.get("n_total") or 0
    n_lo     = int(_get(lo, "n_queries_with_llm_op") or 0)
    avg_lo_in  = _get(lo, "avg_input_tokens")
    avg_lo_out = _get(lo, "avg_output_tokens")
    avg_lo_tok = _get(lo, "avg_total_tokens")
    avg_lo_cr  = _get(lo, "avg_cache_read_tokens") or 0.0
    avg_lo_cw  = _get(lo, "avg_cache_write_tokens") or 0.0

    # LLM-op cost amortised over ALL n_total queries (not just those with calls)
    if avg_lo_in is not None and avg_lo_out is not None and n_lo and n_total:
        avg_lo_non_cached_in = max(0.0, float(avg_lo_in) - float(avg_lo_cr) - float(avg_lo_cw))
        llmop_cost_amortised = (
            (
                avg_lo_non_cached_in * _LLMOP_IN_PRICE
                + float(avg_lo_cw) * _LLMOP_CACHE_WRITE_PRICE
                + float(avg_lo_cr) * _LLMOP_CACHE_READ_PRICE
                + float(avg_lo_out) * _LLMOP_OUT_PRICE
            )
            * (n_lo / n_total)
        )
    else:
        llmop_cost_amortised = 0.0

    avg_agent_cost  = _get(ap, "avg_cost_usd") or 0.0
    avg_agent_total = _get(ap, "avg_total_tokens") or 0.0
    avg_lo_total_amortised = (avg_lo_tok or 0.0) * ((n_lo / n_total) if n_total else 0.0)

    return {
        "success_rate":          _get(perf, "success_rate"),
        "row_level_f1":          _get(perf, "row_level_f1"),
        "item_level_f1":         _get(perf, "item_level_f1"),
        "avg_workflow_length":   _get(lo,   "avg_workflow_length"),
        "avg_elapsed_s":         _get(perf, "avg_elapsed_s"),
        "n_missing_csv":         float(summary.get("n_missing_csv") or 0),
        # Agent planning tokens
        "agent_input_tokens":    _get(ap, "avg_input_tokens"),
        "agent_cache_tokens":    _get(ap, "avg_cache_read_tokens"),
        "agent_output_tokens":   _get(ap, "avg_output_tokens"),
        "agent_total_tokens":    _get(ap, "avg_total_tokens"),
        "agent_cost_per_query":  avg_agent_cost,
        # LLM-op tokens (avg over queries-with-calls; n_queries for context)
        "n_queries_with_llmop":  float(n_lo),
        "llmop_input_tokens":    avg_lo_in,
        "llmop_cache_read_tokens": avg_lo_cr,
        "llmop_cache_write_tokens": avg_lo_cw,
        "llmop_output_tokens":   avg_lo_out,
        "llmop_total_tokens":    avg_lo_tok,
        "llmop_cost_amortised":  llmop_cost_amortised,
        # Combined (amortised over all queries)
        "combined_total_tokens": avg_agent_total + avg_lo_total_amortised,
        "combined_cost_per_query": avg_agent_cost + llmop_cost_amortised,
    }


# ---------------------------------------------------------------------------
# Aggregate across K runs
# ---------------------------------------------------------------------------

def _aggregate(
    metrics_per_run: List[Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, float]]:
    keys = list(metrics_per_run[0].keys())
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [float(m[key]) for m in metrics_per_run if m.get(key) is not None]
        if not vals:
            result[key] = {"mean": float("nan"), "sd": float("nan"), "ci95": float("nan"), "n": 0}
        else:
            result[key] = {
                "mean": _mean(vals),
                "sd":   _std(vals),
                "ci95": _ci95(vals),
                "n":    float(len(vals)),
            }
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: Dict[str, str] = {
    "text2sql":    "Agentic Text2SQL",
    "blendsql":    "Agentic BlendSQL",
    "db_agent":    "DB Agent",
    "db_agent_lx": "DB Agent w/ Learning",
}


def _nan_or(agg: Dict[str, float]) -> tuple:
    m, s = agg.get("mean", float("nan")), agg.get("sd", float("nan"))
    return m, s


def _fmt_pct(agg: Dict[str, float]) -> str:
    m, s = _nan_or(agg)
    if math.isnan(m):
        return "N/A"
    if not math.isnan(s):
        return f"{m:.2f}% +/-{s:.2f}"
    return f"{m:.2f}%"


def _fmt_num(agg: Dict[str, float], fmt: str = ".2f") -> str:
    m, s = _nan_or(agg)
    if math.isnan(m):
        return "N/A"
    if not math.isnan(s):
        return f"{m:{fmt}} +/-{s:{fmt}}"
    return f"{m:{fmt}}"


def _fmt_tok(agg: Dict[str, float]) -> str:
    m, s = _nan_or(agg)
    if math.isnan(m):
        return "N/A"
    if not math.isnan(s):
        return f"{m:,.0f} +/-{s:,.0f}"
    return f"{m:,.0f}"


def _fmt_cost(agg: Dict[str, float]) -> str:
    m, s = _nan_or(agg)
    if math.isnan(m):
        return "N/A"
    if not math.isnan(s):
        return f"${m:.4f} +/-${s:.4f}"
    return f"${m:.4f}"


def _fmt_plain(agg: Dict[str, float], fmt: str = ".1f") -> str:
    m, s = _nan_or(agg)
    if math.isnan(m):
        return "N/A"
    if not math.isnan(s):
        return f"{m:{fmt}} +/-{s:{fmt}}"
    return f"{m:{fmt}}"


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(
    agents: List[str],
    agg_per_agent: Dict[str, Dict[str, Dict[str, float]]],
    k: int,
    n_total: Optional[int],
) -> None:
    labels = [_DISPLAY_NAMES.get(s, s) for s in agents]
    header_cols = ["Metric"] + [f"{lab} (K={k})" for lab in labels]
    # Dynamic column width: at least 45 for first, 26 for others
    col_widths = [45] + [max(26, len(h) + 2) for h in header_cols[1:]]

    def row(*cells: str) -> str:
        padded = []
        for i, c in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 20
            padded.append(c.ljust(w))
        return "| " + " | ".join(padded) + " |"

    def sep() -> str:
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    bench = f"{n_total} queries" if n_total else "?"
    print(f"\n## SWAN Multi-Run Comparison ({k} repetitions) -- {bench}\n")
    print(f"Values shown as mean +/- SD (sample std dev, K={k} runs).\n")
    print(row(*header_cols))
    print(sep())

    def mrow(label: str, key: str, fmt_fn) -> None:
        cells = [label]
        for slug in agents:
            agg_m = (agg_per_agent.get(slug) or {}).get(
                key, {"mean": float("nan"), "sd": float("nan")}
            )
            cells.append(fmt_fn(agg_m))
        print(row(*cells))

    def blank() -> None:
        print(row(*[""] * len(header_cols)))

    mrow("**Success Rate**",           "success_rate",       _fmt_pct)
    mrow("**Row-level F1**",           "row_level_f1",       _fmt_pct)
    mrow("**Item-level F1**",          "item_level_f1",      _fmt_pct)
    mrow("**Avg workflow length**",    "avg_workflow_length",
         lambda a: _fmt_num(a, fmt=".2f"))
    mrow("**Avg walltime/query (s)**", "avg_elapsed_s",
         lambda a: _fmt_num(a, fmt=".0f"))
    mrow("**Missing CSV / failed**",   "n_missing_csv",
         lambda a: _fmt_plain(a, fmt=".1f"))
    blank()
    print(row("**Agent planning tokens (avg)**", *[""] * len(agents)))
    mrow("-- Input",                   "agent_input_tokens",   _fmt_tok)
    mrow("-- Cached input",            "agent_cache_tokens",   _fmt_tok)
    mrow("-- Output",                  "agent_output_tokens",  _fmt_tok)
    mrow("-- Total",                   "agent_total_tokens",   _fmt_tok)
    mrow("Avg cost/query",             "agent_cost_per_query", _fmt_cost)
    blank()
    print(row("**LLM-op tokens (avg, queries with calls)**", *[""] * len(agents)))
    mrow("-- Queries with calls",      "n_queries_with_llmop",
         lambda a: _fmt_plain(a, fmt=".1f"))
    mrow("-- Input",                   "llmop_input_tokens",   _fmt_tok)
    mrow("-- Output",                  "llmop_output_tokens",  _fmt_tok)
    mrow("-- Total",                   "llmop_total_tokens",   _fmt_tok)
    mrow("Avg LLM-op cost/query",      "llmop_cost_amortised", _fmt_cost)
    blank()
    mrow("**Combined avg total tokens**", "combined_total_tokens",    _fmt_tok)
    mrow("**Combined avg cost/query**",   "combined_cost_per_query",  _fmt_cost)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate K SWAN eval runs and print comparison table."
    )
    p.add_argument("--base_dir", required=True)
    p.add_argument(
        "--agents", nargs="+",
        default=["text2sql", "blendsql", "db_agent", "db_agent_lx"],
    )
    p.add_argument("--k", type=int, required=True)
    args = p.parse_args(argv)

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = (Path(__file__).resolve().parent / base_dir).resolve()

    agents: List[str] = args.agents
    k = args.k
    agg_per_agent: Dict[str, Dict[str, Dict[str, float]]] = {}
    n_total_global: Optional[int] = None

    for slug in agents:
        metrics_list: List[Dict[str, Optional[float]]] = []
        missing: List[int] = []

        for i in range(1, k + 1):
            spath = base_dir / slug / f"run_{i}" / "eval_summary.json"
            summary = _load_summary(spath)
            if summary is None:
                print(f"  [WARN] Missing: {spath}", file=sys.stderr)
                missing.append(i)
                continue
            if n_total_global is None:
                n_total_global = summary.get("n_total")
            metrics_list.append(_extract_metrics(summary))

        if missing:
            print(f"  [WARN] {slug}: runs {missing} missing ({len(missing)}/{k})", file=sys.stderr)

        if not metrics_list:
            print(f"  [ERROR] No valid runs found for agent {slug!r}.", file=sys.stderr)
            agg_per_agent[slug] = {}
            continue

        agg_per_agent[slug] = _aggregate(metrics_list)
        print(f"  {slug}: {len(metrics_list)}/{k} runs aggregated.", file=sys.stderr)

    _print_table(agents, agg_per_agent, k=k, n_total=n_total_global)

    # Save JSON (NaN -> null)
    def _clean(v: float) -> Optional[float]:
        return None if math.isnan(v) else v

    out: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "k": k,
        "agents": agents,
        "n_total": n_total_global,
        "aggregate": {
            slug: {
                metric: {stat: _clean(float(val)) for stat, val in stats.items()}
                for metric, stats in agg.items()
            }
            for slug, agg in agg_per_agent.items()
        },
    }
    out_path = base_dir / "aggregate_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aggregate summary saved to: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
