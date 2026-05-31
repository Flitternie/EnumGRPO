"""
General-purpose evaluation utilities for SWAN-style table answers.

This module is designed to be reusable regardless of how predictions are loaded/stored
(CSV, JSON, in-memory lists, etc.). The core APIs operate on **rows** (list-of-lists)
and provide three granularities:

- **Success Rate (SR)**: binary perfect-match metric
- **Row-level PRF**: compares rows as units (unordered multiset by default; ordered when required)
- **Item-level PRF**: compares individual cells/items (order-invariant)

It also includes optional SWAN helpers:
- `GroundTruth` + `load_ground_truth_jsonl`
- `requires_ordering_from_text` / `requires_ordering`
- agent-run parsing for LLM-operator usage (tools named `llm_*`)
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

DEFAULT_NUM_DECIMALS = 2

CanonCell = tuple[str, str]
CanonRow = tuple[CanonCell | None, ...]

__all__ = [
    # SWAN helpers
    "GroundTruth",
    "load_ground_truth_jsonl",
    "requires_ordering_from_text",
    "requires_ordering",
    # Canonicalization / table helpers
    "table_ncols",
    "pad_rows",
    "canon_cell",
    "canon_row",
    "canon_table",
    # Scoring
    "prf_from_counters",
    "score_success_rate",
    "score_row_level",
    "score_item_level",
    # Alignment + evaluation
    "evaluate_tables",
    "evaluate_ground_truth",
]


# -----------------------------
# SWAN ground truth helpers
# -----------------------------


@dataclass(frozen=True)
class GroundTruth:
    """One SWAN question's ground-truth record.

    The `answer_rows` field matches `swan/swan.jsonl`'s `answer` field: list of rows,
    each row a list of cell values.

    For tie-at-boundary queries, `answer_rows_alts` holds additional valid answer sets
    (from `answer_b`, `answer_c`, … in the JSONL). A prediction is correct if it matches
    `answer_rows` OR any entry in `answer_rows_alts`.
    """

    question_id: str
    db: str
    query: str
    sql: str
    answer_rows: list[list[Any]]
    answer_rows_alts: list[list[list[Any]]] = dataclasses.field(default_factory=list)
    required_columns: list[str] = dataclasses.field(default_factory=list)


def load_ground_truth_jsonl(jsonl_path: Path) -> dict[str, GroundTruth]:
    """Load SWAN ground truth from `swan.jsonl` (JSONL format)."""

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {jsonl_path}")

    out: dict[str, GroundTruth] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id")
            if not qid:
                raise ValueError(f"Missing question_id at line {line_no}")

            # Collect answer_b, answer_c, … as alternative valid answer sets.
            alts: list[list[list[Any]]] = []
            for key in sorted(k for k in rec if k.startswith("answer_")):
                val = rec[key]
                if isinstance(val, list):
                    alts.append(val)

            out[qid] = GroundTruth(
                question_id=str(qid),
                db=str(rec.get("db", "")),
                query=str(rec.get("query", "")),
                sql=str(rec.get("sql", "")),
                answer_rows=list(rec.get("answer") or []),
                answer_rows_alts=alts,
                required_columns=list(rec.get("required_columns") or []),
            )
    return out


# -----------------------------
# Ordering heuristics
# -----------------------------


ORDER_HINT_RE = re.compile(
    r"\b(order\s+by|sort|sorted|ascending|descending|top\s+\d+|lowest\s+\d+|highest\s+\d+)\b",
    re.IGNORECASE,
)


def requires_ordering_from_text(query: str | None, sql: str | None) -> bool:
    """Heuristic to decide whether results should be evaluated with ordering."""

    q = (query or "")
    s = (sql or "")
    return ("order by" in s.casefold()) or bool(ORDER_HINT_RE.search(q))


def requires_ordering(gt_rec: GroundTruth) -> bool:
    """Convenience wrapper around `requires_ordering_from_text` for a `GroundTruth` record."""

    return requires_ordering_from_text(gt_rec.query, gt_rec.sql)


# -----------------------------
# Canonicalization
# -----------------------------


_NUM_LIKE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _quantize_decimal(d: Decimal, decimals: int) -> Decimal:
    if decimals <= 0:
        return d.to_integral_value(rounding=ROUND_HALF_UP)
    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    return d.quantize(q, rounding=ROUND_HALF_UP)


def canon_cell(x: Any, *, num_decimals: int = DEFAULT_NUM_DECIMALS) -> CanonCell | None:
    """Canonicalize a cell for robust comparisons.

    - Null-ish values -> None
    - Numeric-like strings -> ("num", normalized_string) with rounding
    - Everything else -> ("str", casefolded_string)
    """

    if x is None:
        return None

    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None

    s = str(x).strip()
    if s == "" or s.casefold() in {"null", "none", "nan"}:
        return None

    if _NUM_LIKE_RE.match(s):
        try:
            d = Decimal(s)
        except InvalidOperation:
            return ("str", s.casefold())

        d = _quantize_decimal(d, num_decimals)
        s2 = format(d, "f")
        if "." in s2:
            s2 = s2.rstrip("0").rstrip(".")
        if s2 == "-0":
            s2 = "0"
        return ("num", s2)

    return ("str", s.casefold())


def canon_row(row: Iterable[Any], *, num_decimals: int = DEFAULT_NUM_DECIMALS) -> CanonRow:
    """Canonicalize a row (sequence of cells)."""

    return tuple(canon_cell(v, num_decimals=num_decimals) for v in row)


def canon_table(rows: Sequence[Sequence[Any]], *, num_decimals: int = DEFAULT_NUM_DECIMALS) -> list[CanonRow]:
    """Canonicalize a table represented as a list of rows."""

    return [canon_row(r, num_decimals=num_decimals) for r in rows]


# -----------------------------
# Table shape / projection helpers
# -----------------------------


def table_ncols(rows: Sequence[Sequence[Any]]) -> int:
    """Max row length in a possibly ragged table."""

    return max((len(r) for r in rows), default=0)


def pad_rows(rows: Sequence[Sequence[Any]], ncols: int, pad_value: Any = None) -> list[list[Any]]:
    """Pad each row to `ncols` columns (ragged -> rectangular)."""

    out: list[list[Any]] = []
    for r in rows:
        rr = list(r)
        if len(rr) < ncols:
            rr = rr + [pad_value] * (ncols - len(rr))
        out.append(rr)
    return out


def _drop_all_none_rows(rows: list[CanonRow]) -> list[CanonRow]:
    return [r for r in rows if any(v is not None for v in r)]


# -----------------------------
# Scoring primitives
# -----------------------------


def prf_from_counters(gt: Counter, pred: Counter) -> dict[str, float]:
    """Precision/Recall/F1 from two multisets (Counter).

    Both empty → perfect match (all metrics = 1.0).
    gt non-empty, pred empty → recall = 0, f1 = 0.
    gt empty, pred non-empty → precision = 0, f1 = 0.
    """
    tp = sum(min(gt[k], pred[k]) for k in (gt.keys() & pred.keys()))
    gt_n = sum(gt.values())
    pred_n = sum(pred.values())

    # Both sides are empty: this is a perfect prediction of an empty result set.
    if gt_n == 0 and pred_n == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0.0, "gt_n": 0.0, "pred_n": 0.0}

    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "gt_n": float(gt_n), "pred_n": float(pred_n)}


def score_success_rate(gt_rows_can_all: list[CanonRow], pred_rows_can_all: list[CanonRow], ordered: bool) -> dict[str, Any]:
    """Binary SR (perfect match), after fixed column alignment."""

    if ordered:
        return {"sr": int(gt_rows_can_all == pred_rows_can_all)}

    return {"sr": int(Counter(_drop_all_none_rows(gt_rows_can_all)) == Counter(_drop_all_none_rows(pred_rows_can_all)))}


def score_row_level(gt_rows_can_all: list[CanonRow], pred_rows_can_all: list[CanonRow], ordered: bool) -> dict[str, Any]:
    """Row-level PRF."""

    if ordered:
        # Both empty → perfect match.
        if not gt_rows_can_all and not pred_rows_can_all:
            return {
                "row_precision": 1.0, "row_recall": 1.0, "row_f1": 1.0,
                "row_tp": 0.0, "row_gt_n": 0.0, "row_pred_n": 0.0,
            }
        correct = sum(
            1
            for i in range(min(len(gt_rows_can_all), len(pred_rows_can_all)))
            if gt_rows_can_all[i] == pred_rows_can_all[i]
        )
        precision = correct / len(pred_rows_can_all) if pred_rows_can_all else 0.0
        recall = correct / len(gt_rows_can_all) if gt_rows_can_all else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "row_precision": precision,
            "row_recall": recall,
            "row_f1": f1,
            "row_tp": float(correct),
            "row_gt_n": float(len(gt_rows_can_all)),
            "row_pred_n": float(len(pred_rows_can_all)),
        }

    m = prf_from_counters(Counter(_drop_all_none_rows(gt_rows_can_all)), Counter(_drop_all_none_rows(pred_rows_can_all)))
    return {
        "row_precision": m["precision"],
        "row_recall": m["recall"],
        "row_f1": m["f1"],
        "row_tp": m["tp"],
        "row_gt_n": m["gt_n"],
        "row_pred_n": m["pred_n"],
    }


def score_item_level(gt_rows_can_all: list[CanonRow], pred_rows_can_all: list[CanonRow]) -> dict[str, Any]:
    """Item-level PRF over canonicalized cells (NULL-ish ignored)."""

    def flatten_items(rows: list[CanonRow]) -> Counter:
        items: list[CanonCell] = []
        for r in rows:
            for c in r:
                if c is not None:
                    items.append(c)
        return Counter(items)

    m = prf_from_counters(flatten_items(gt_rows_can_all), flatten_items(pred_rows_can_all))
    return {
        "item_precision": m["precision"],
        "item_recall": m["recall"],
        "item_f1": m["f1"],
        "item_tp": m["tp"],
        "item_gt_n": m["gt_n"],
        "item_pred_n": m["pred_n"],
    }


# -----------------------------
# Column alignment + evaluation
# -----------------------------


def _project_pred_rows(pred_rows: Sequence[Sequence[Any]], mapping: tuple[int, ...], gt_cols: int) -> list[list[Any]]:
    """Project prediction rows to match gold column order, then pad to gt_cols."""

    out: list[list[Any]] = []
    for r in pred_rows:
        rr = list(r)
        proj = [rr[i] if 0 <= i < len(rr) else None for i in mapping]
        if len(proj) < gt_cols:
            proj = proj + [None] * (gt_cols - len(proj))
        out.append(proj)
    return out


def evaluate_tables(
    gt_rows_raw: Sequence[Sequence[Any]],
    pred_rows_raw: Sequence[Sequence[Any]],
    *,
    ordered: bool,
    num_decimals: int = DEFAULT_NUM_DECIMALS,
    max_total_mappings: int = 200,
) -> dict[str, Any]:
    """Evaluate predicted table rows against gold table rows.

    This function is agnostic to where predictions come from; just pass the rows.

    Column alignment:
      - If pred has >= gold columns: search a single best fixed mapping (subset+permutation)
        chosen by maximizing row-level F1.
      - If pred has < gold columns: pad missing columns with NULLs (no alignment search).
    """

    gt_cols = table_ncols(gt_rows_raw)
    pred_cols = table_ncols(pred_rows_raw)
    gt_rows_pad = pad_rows(gt_rows_raw, gt_cols)

    gt_can = canon_table(gt_rows_pad, num_decimals=num_decimals)

    # Build aligned prediction canonical rows.
    chosen_mapping: tuple[int, ...]
    mappings_checked = 0

    if pred_cols < gt_cols:
        pred_pad = pad_rows(pred_rows_raw, gt_cols)
        pred_can = canon_table(pred_pad, num_decimals=num_decimals)
        chosen_mapping = tuple(range(pred_cols))
    else:
        best_row = None
        best_mapping = ()
        best_pred_can: list[CanonRow] = []

        def better(a: dict[str, Any], b: dict[str, Any]) -> bool:
            return (a["row_f1"], a["row_recall"], a["row_precision"]) > (b["row_f1"], b["row_recall"], b["row_precision"])

        for combo in itertools.combinations(range(pred_cols), gt_cols):
            for mapping in itertools.permutations(combo):
                pred_proj = _project_pred_rows(pred_rows_raw, tuple(mapping), gt_cols)
                pred_can_tmp = canon_table(pred_proj, num_decimals=num_decimals)
                row_m = score_row_level(gt_can, pred_can_tmp, ordered)

                if best_row is None or better(row_m, best_row):
                    best_row = row_m
                    best_mapping = tuple(mapping)
                    best_pred_can = pred_can_tmp

                mappings_checked += 1
                if mappings_checked >= max_total_mappings:
                    break
            if mappings_checked >= max_total_mappings:
                break

        chosen_mapping = best_mapping
        pred_can = best_pred_can

    out: dict[str, Any] = {
        "gt_cols": gt_cols,
        "pred_cols": pred_cols,
        "gt_rows": len(gt_rows_raw),
        "pred_rows": len(pred_rows_raw),
        "chosen_pred_cols": chosen_mapping,
    }
    if pred_cols >= gt_cols:
        out["mappings_checked"] = mappings_checked

    out.update(score_success_rate(gt_can, pred_can, ordered))
    out.update(score_row_level(gt_can, pred_can, ordered))
    out.update(score_item_level(gt_can, pred_can))

    # Back-compat aliases for older notebooks.
    out["precision"] = float(out.get("row_precision", 0.0))
    out["recall"] = float(out.get("row_recall", 0.0))
    out["f1"] = float(out.get("row_f1", 0.0))
    out["tp"] = float(out.get("row_tp", 0.0))
    out["gt_n"] = float(out.get("row_gt_n", 0.0))
    out["pred_n"] = float(out.get("row_pred_n", 0.0))

    return out


def evaluate_ground_truth(
    gt_rec: GroundTruth,
    pred_rows_raw: Sequence[Sequence[Any]],
    *,
    num_decimals: int = DEFAULT_NUM_DECIMALS,
    max_total_mappings: int = 200,
    force_unordered: bool = False,
) -> dict[str, Any]:
    """Evaluate a `GroundTruth` record against predicted rows.

    When the ground truth has alternative valid answer sets (tie-at-boundary
    queries), the prediction is evaluated against every eligible answer and the
    best result is returned.  The output includes `num_answers_tried` so callers
    can tell whether a tie-aware comparison was performed.

    Pass ``force_unordered=True`` to ignore the ordering heuristic and always
    compare rows as an unordered multiset.
    """

    ordered = False if force_unordered else requires_ordering(gt_rec)

    # Build the full list of eligible answer sets: primary first, then alts.
    all_answer_sets: list[list[list[Any]]] = [gt_rec.answer_rows or []]
    all_answer_sets.extend(gt_rec.answer_rows_alts)

    best_out: dict[str, Any] | None = None
    for answer_rows in all_answer_sets:
        candidate = evaluate_tables(
            answer_rows,
            pred_rows_raw,
            ordered=ordered,
            num_decimals=num_decimals,
            max_total_mappings=max_total_mappings,
        )
        # Select the answer version that gives the best combined performance.
        # ALL metrics (sr, row_f1, item_f1, …) are taken from the single chosen
        # candidate — never mixed across different answer versions.
        if best_out is None or (
            candidate["sr"],
            candidate["row_f1"],
            candidate["item_f1"],
        ) > (
            best_out["sr"],
            best_out["row_f1"],
            best_out["item_f1"],
        ):
            best_out = candidate

    assert best_out is not None
    best_out["question_id"] = gt_rec.question_id
    best_out["db"] = gt_rec.db
    best_out["requires_order"] = ordered
    best_out["num_answers_tried"] = len(all_answer_sets)
    return best_out
