#!/usr/bin/env python3
"""Evaluate an agent run with SemBench Movie's query-specific metrics.

The metric behavior follows SemBench revision
c814e3807e72d4cf876b852b17e77f3cc94575c2. This standalone adapter avoids
pulling SemBench's unrelated multimodal evaluator dependencies into the agent
environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_swan import (
    _LLMOP_CACHE_READ_PRICE_PER_TOK,
    _LLMOP_CACHE_WRITE_PRICE_PER_TOK,
    _LLMOP_INPUT_PRICE_PER_TOK,
    _LLMOP_OUTPUT_PRICE_PER_TOK,
    _extract_agent_token_usage,
    _extract_mcp_tool_stats,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _retrieval_limit(pred: pd.DataFrame, gold: pd.DataFrame, limit: int) -> dict[str, float]:
    if pred.empty:
        precision = 1.0 if gold.empty else 0.0
        return {"precision": precision, "recall": 0.0, "f1_score": 0.0}
    if gold.empty or len(pred.columns) == 0 or len(gold.columns) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    pred_ids = set(pred.head(limit).iloc[:, 0].dropna())
    gold_ids = set(gold.iloc[:, 0].dropna())
    correct = pred_ids & gold_ids
    precision = len(correct) / len(pred_ids) if pred_ids else 0.0
    recall = min(len(correct), limit) / min(limit, len(gold_ids)) if gold_ids else 0.0
    return {"precision": precision, "recall": recall, "f1_score": _f1(precision, recall)}


def _normalized_pairs(frame: pd.DataFrame) -> set[tuple[Any, tuple[Any, Any]]]:
    if len(frame.columns) < 3:
        return set()
    output: set[tuple[Any, tuple[Any, Any]]] = set()
    for row in frame.itertuples(index=False, name=None):
        movie, left, right = row[:3]
        if pd.isna(movie) or pd.isna(left) or pd.isna(right):
            continue
        output.add((movie, tuple(sorted((left, right), key=str))))
    return output


def _pair_retrieval(pred: pd.DataFrame, gold: pd.DataFrame, limit: int | None) -> dict[str, float]:
    if pred.empty:
        precision = 1.0 if gold.empty else 0.0
        return {"precision": precision, "recall": 0.0, "f1_score": 0.0}
    if gold.empty or len(pred.columns) < 3 or len(gold.columns) < 3:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    if limit is not None:
        pred = pred.head(limit)
    pred_pairs = _normalized_pairs(pred)
    gold_pairs = _normalized_pairs(gold)
    correct = pred_pairs & gold_pairs
    precision = len(correct) / len(pred_pairs) if pred_pairs else 0.0
    denominator = min(limit, len(gold_pairs)) if limit is not None else len(gold_pairs)
    recall = min(len(correct), limit) / denominator if limit is not None and denominator else (
        len(correct) / denominator if denominator else 0.0
    )
    return {"precision": precision, "recall": recall, "f1_score": _f1(precision, recall)}


def _first_number(frame: pd.DataFrame) -> float | None:
    if len(frame) != 1:
        return None
    for value in frame.iloc[0]:
        number = _safe_float(value)
        if number is not None:
            return number
    return None


def _aggregation(pred: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float | None]:
    predicted = _first_number(pred)
    actual = _first_number(gold)
    if predicted is None or actual is None:
        return {"relative_error": 1.0, "absolute_error": None, "mean_absolute_percentage_error": 100.0}
    absolute = abs(predicted - actual)
    relative = absolute / abs(actual) if actual else (0.0 if predicted == 0 else math.inf)
    return {
        "relative_error": relative if math.isfinite(relative) else None,
        "absolute_error": absolute,
        "mean_absolute_percentage_error": relative * 100.0 if math.isfinite(relative) else None,
    }


def _sentiment_counts(pred: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float | None]:
    if pred.empty or gold.empty or len(pred.columns) < 2 or len(gold.columns) < 2:
        return {"relative_error": 1.0, "absolute_error": None, "mean_absolute_percentage_error": 100.0}

    def counts(frame: pd.DataFrame) -> dict[str, float]:
        result: dict[str, float] = {}
        for label, value, *_ in frame.itertuples(index=False, name=None):
            number = _safe_float(value)
            if pd.notna(label) and number is not None:
                result[str(label).strip().upper()] = number
        return result

    predicted, actual = counts(pred), counts(gold)
    if not predicted or not actual:
        return {"relative_error": 1.0, "absolute_error": None, "mean_absolute_percentage_error": 100.0}
    absolute_total = 0.0
    relative_errors: list[float] = []
    for label in set(predicted) | set(actual):
        p, g = predicted.get(label, 0.0), actual.get(label, 0.0)
        absolute_total += abs(p - g)
        if g:
            relative_errors.append(abs(p - g) / abs(g))
        elif p:
            relative_errors.append(1.0)
    relative = sum(relative_errors) / len(relative_errors) if relative_errors else 0.0
    return {
        "relative_error": relative,
        "absolute_error": absolute_total,
        "mean_absolute_percentage_error": relative * 100.0,
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[ordered[pos]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    lmean, rmean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - lmean) * (y - rmean) for x, y in zip(left, right))
    lnorm = sum((x - lmean) ** 2 for x in left)
    rnorm = sum((y - rmean) ** 2 for y in right)
    denominator = math.sqrt(lnorm * rnorm)
    return numerator / denominator if denominator else 0.0


def _kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = (left[i] > left[j]) - (left[i] < left[j])
            dy = (right[i] > right[j]) - (right[i] < right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) *
        (concordant + discordant + ties_right)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def _ranking(pred: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float | int]:
    if pred.empty or gold.empty or len(pred.columns) < 2 or len(gold.columns) < 2:
        return {"spearman_correlation": 0.0, "kendall_tau": 0.0, "common_ids": 0}

    def scores(frame: pd.DataFrame) -> dict[Any, float]:
        result: dict[Any, float] = {}
        for key, value, *_ in frame.itertuples(index=False, name=None):
            number = _safe_float(value)
            if pd.notna(key) and number is not None:
                result[key] = number
        return result

    predicted, actual = scores(pred), scores(gold)
    common = sorted(set(predicted) & set(actual), key=str)
    if len(common) < 2:
        return {"spearman_correlation": 0.0, "kendall_tau": 0.0, "common_ids": len(common)}
    pvals = [predicted[key] for key in common]
    gvals = [actual[key] for key in common]
    return {
        "spearman_correlation": _pearson(_average_ranks(pvals), _average_ranks(gvals)),
        "kendall_tau": _kendall_tau_b(pvals, gvals),
        "common_ids": len(common),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _load_run_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "results.jsonl"
    output: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            qid = str(record.get("job", {}).get("question_id", ""))
            elapsed = _safe_float(record.get("elapsed_s"))
            stderr = str(record.get("stderr") or "")
            if qid:
                output[qid] = {
                    "elapsed_s": elapsed,
                    "runner_ok": bool(record.get("ok")),
                    "iteration_limit": "Agent reached maximum" in stderr,
                }
        except Exception:
            continue
    return output


def main() -> int:
    args = _args()
    run_dir = args.run_dir.resolve()
    prepared = args.prepared_root.resolve()
    run_records = _load_run_records(run_dir)
    rows: list[dict[str, Any]] = []

    for qid in range(1, 11):
        key = f"Q{qid}"
        pred_path = run_dir / f"{key}.csv"
        gold_path = prepared / "ground_truth" / f"{key}.csv"
        gold = pd.read_csv(gold_path)
        run_record = run_records.get(key, {})
        missing = not pred_path.is_file()
        iteration_limit = bool(run_record.get("iteration_limit"))
        runner_failed = bool(run_record) and not bool(run_record.get("runner_ok"))
        invalid = missing or iteration_limit or runner_failed
        try:
            pred = pd.read_csv(pred_path) if not invalid else pd.DataFrame()
        except pd.errors.EmptyDataError:
            pred = pd.DataFrame()

        if qid in (1, 2):
            metric_type, metrics = "retrieval", _retrieval_limit(pred, gold, 5)
        elif qid in (5, 6):
            metric_type, metrics = "retrieval", _pair_retrieval(pred, gold, 10)
        elif qid == 7:
            metric_type, metrics = "retrieval", _pair_retrieval(pred, gold, None)
        elif qid in (3, 4):
            metric_type, metrics = "aggregation", _aggregation(pred, gold)
        elif qid == 8:
            metric_type, metrics = "aggregation", _sentiment_counts(pred, gold)
        else:
            metric_type, metrics = "ranking", _ranking(pred, gold)

        planner = _extract_agent_token_usage(run_dir, key)
        llmop = _extract_mcp_tool_stats(run_dir, key)
        usage = llmop.get("llm_op_token_usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read_tokens = int(usage.get("cache_read_tokens", 0))
        cache_write_tokens = int(usage.get("cache_write_tokens", 0))
        non_cached_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
        llm_op_cost = (
            non_cached_input * _LLMOP_INPUT_PRICE_PER_TOK
            + cache_read_tokens * _LLMOP_CACHE_READ_PRICE_PER_TOK
            + cache_write_tokens * _LLMOP_CACHE_WRITE_PRICE_PER_TOK
            + output_tokens * _LLMOP_OUTPUT_PRICE_PER_TOK
        )
        rows.append(
            {
                "query_id": key,
                "metric_type": metric_type,
                "missing_output": missing,
                "iteration_limit": iteration_limit,
                "runner_failed": runner_failed,
                "valid_output": not invalid,
                "prediction_rows": int(len(pred)),
                "ground_truth_rows": int(len(gold)),
                "metrics": metrics,
                "elapsed_s": run_record.get("elapsed_s"),
                "planner_cost_usd": planner.get("agent_cost_usd", 0.0),
                "llm_op_calls": int(llmop.get("llm_op_calls", 0)),
                "workflow_length": int(llmop.get("workflow_length", 0)),
                "llm_op_token_usage": usage,
                "llm_op_cost_usd": llm_op_cost,
            }
        )

    retrieval = [row for row in rows if row["metric_type"] == "retrieval"]
    aggregation = [row for row in rows if row["metric_type"] == "aggregation"]
    ranking = [row for row in rows if row["metric_type"] == "ranking"]
    summary = {
        "benchmark": "SemBench",
        "scenario": "movie",
        "scale_factor": 2000,
        "n_queries": 10,
        "n_missing_outputs": sum(bool(row["missing_output"]) for row in rows),
        "n_iteration_limit": sum(bool(row["iteration_limit"]) for row in rows),
        "n_failed_tasks": sum(not bool(row["valid_output"]) for row in rows),
        "official_metric_means": {
            "retrieval_f1": _mean(row["metrics"].get("f1_score") for row in retrieval),
            "aggregation_relative_error": _mean(row["metrics"].get("relative_error") for row in aggregation),
            "ranking_spearman": _mean(row["metrics"].get("spearman_correlation") for row in ranking),
            "ranking_kendall_tau": _mean(row["metrics"].get("kendall_tau") for row in ranking),
        },
        "runtime": {
            "total_elapsed_s": sum(row["elapsed_s"] or 0.0 for row in rows),
            "mean_elapsed_s": _mean(row["elapsed_s"] for row in rows),
            "total_planner_cost_usd": sum(float(row["planner_cost_usd"] or 0.0) for row in rows),
            "total_llm_op_cost_usd": sum(float(row["llm_op_cost_usd"] or 0.0) for row in rows),
            "total_llm_op_calls": sum(int(row["llm_op_calls"]) for row in rows),
        },
        "queries": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
