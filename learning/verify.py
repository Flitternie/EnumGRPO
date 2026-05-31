"""Multi-objective verify_func for training-free GRPO.

Each rollout produces a trajectory for one query.  This module scores it on:

  - Performance  (how correct is the final answer?)
      sr       — binary success rate (1.0 = exact match, 0.0 = wrong)
      row_f1   — row-level F1 against ground-truth rows
      item_f1  — item-level F1 against ground-truth cells

  - Cost  (how expensive was the trajectory?)
      llmop_tokens — total tokens consumed by LLM operator tools
                     (llm_map / llm_reduce / run_blendsql) in this rollout

The scalar ``reward`` returned by ``verify_func`` is performance-only:

    perf = w_sr * sr + w_row * row_f1 + w_item * item_f1   ∈ [0, 1]

Cost is NOT folded in here.  Instead ``llmop_tokens`` is returned as a
separate field so the experience updater can normalise it **within the group
for that query** before applying the cost penalty:

    cost_rank = (tokens - min_tokens_in_group)
                / max(max_tokens - min_tokens, 1)   ∈ [0, 1]

    final_reward = perf - w_cost * cost_rank

This ensures:
  • Performance always dominates — two rollouts with different perf scores
    are ranked by performance regardless of cost.
  • Cost is only a tiebreaker — among rollouts with equal perf, the cheaper
    one wins by at most ``w_cost``.
  • Cost is query-relative — token counts are normalised within the group,
    so an inherently expensive query is not penalised against a cheap one.

Weights are loaded from ``EvalConfig`` (populated by ``learning/config.yaml``)
and injected into every ``verify_func`` call via the ``_weights`` key.
Calling ``verify_func`` without ``_weights`` raises ``KeyError``.

Answer parsing
--------------
The agent returns a plain-text ``final_answer``.  We try (in order):

  1. JSON array-of-arrays  (already structured)
  2. Markdown / ASCII table  (| col | col | delimiters)
  3. CSV block inside a code fence
  4. Plain CSV  (comma-separated lines)
  5. Single scalar value  (wrapped as [[value]])

Any parse failure falls back to an empty row list → sr=0, f1=0.
An empty ground-truth row list means the expected answer is zero rows;
only an empty prediction matches (score = 1.0).  A non-empty prediction
against an empty ground truth scores 0.0.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV output reader
# ---------------------------------------------------------------------------

def read_csv_rows(path: str | Path) -> list[list[Any]] | None:
    """Read a CSV file written by the agent and return it as a list of rows.

    The first row is assumed to be a header and is skipped.
    Returns ``None`` if the file does not exist or cannot be parsed.
    Numeric strings are coerced to ``int`` / ``float`` so they compare
    correctly against numeric ground-truth values.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("Output CSV not found: %s", p)
        return None
    try:
        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(c.strip() for c in r)]
        if not rows:
            return []
        # Drop header row if present (first row is always treated as header)
        data_rows = rows[1:] if len(rows) > 1 else rows
        # Coerce numeric strings
        coerced: list[list[Any]] = []
        for row in data_rows:
            new_row: list[Any] = []
            for cell in row:
                cell = cell.strip()
                try:
                    new_row.append(int(cell))
                    continue
                except ValueError:
                    pass
                try:
                    new_row.append(float(cell))
                    continue
                except ValueError:
                    pass
                new_row.append(cell)
            coerced.append(new_row)
        return coerced
    except Exception as exc:
        logger.warning("Failed to read output CSV %s: %s", p, exc)
        return None


# ---------------------------------------------------------------------------
# Weight / cost helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|[-:|\s]+\|\s*$")


def _parse_json_rows(text: str) -> list[list[Any]] | None:
    """Try to decode text as JSON array-of-arrays (or array-of-scalars)."""
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, list):
        return None
    rows: list[list[Any]] = []
    for item in obj:
        if isinstance(item, list):
            rows.append(item)
        else:
            rows.append([item])
    return rows or None


def _parse_markdown_table(text: str) -> list[list[Any]] | None:
    """Parse Markdown/ASCII table delimited by | characters (skips separator rows)."""
    lines = [l for l in text.splitlines() if "|" in l]
    if not lines:
        return None
    rows: list[list[Any]] = []
    for line in lines:
        if _MD_TABLE_SEP_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    # First row is typically the header — drop it.
    data = rows[1:] if len(rows) > 1 else rows
    return data or None


def _parse_csv_block(text: str) -> list[list[Any]] | None:
    """Parse comma-separated lines (with or without a code fence)."""
    # Extract from code fence if present.
    m = _FENCE_RE.search(text)
    block = m.group(1) if m else text
    try:
        reader = csv.reader(io.StringIO(block.strip()))
        rows = [r for r in reader if any(c.strip() for c in r)]
    except csv.Error:
        return None
    if not rows:
        return None
    # If first row looks like a header (all strings, none numeric) and there are more
    # rows, drop it — ground truth rows don't include headers.
    if len(rows) > 1:
        rows = rows[1:]
    return rows or None


def parse_answer_rows(text: str) -> list[list[Any]]:
    """
    Parse the agent's free-text final_answer into a list of rows.

    Returns an empty list if the answer cannot be parsed.
    """
    if not text or not text.strip():
        return []

    # 1. JSON array-of-arrays
    rows = _parse_json_rows(text.strip())
    if rows is not None:
        return rows

    # 2. Markdown table
    rows = _parse_markdown_table(text)
    if rows is not None:
        return rows

    # 3. CSV (possibly inside a code fence)
    rows = _parse_csv_block(text)
    if rows is not None and len(rows) > 0 and len(rows[0]) > 1:
        return rows

    # 4. Single scalar — wrap as [[value]]
    stripped = text.strip()
    if "\n" not in stripped and "," not in stripped:
        return [[stripped]]

    # 5. CSV fallback even for single-column
    if rows is not None:
        return rows

    return []


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_answer(
    pred_rows: Sequence[Sequence[Any]],
    gt_rows: Sequence[Sequence[Any]],
    *,
    gt_rows_alts: list[list[list[Any]]] | None = None,
    question: str = "",
    num_decimals: int = 2,
) -> dict[str, float]:
    """Compute SR, row-F1 and item-F1 via ``evaluate_ground_truth``.

    Delegates entirely to ``swan.evaluation.utils.evaluate_ground_truth`` so
    alternative answer sets, numeric tolerance, and ordering are all handled
    identically to the production evaluator (``eval_swan.py``).

    *question* is used for ordering detection (same heuristic as eval_swan).
    """
    from swan.evaluation.utils import GroundTruth, evaluate_ground_truth

    gt_rec = GroundTruth(
        question_id="",
        db="",
        query=question,   # drives requires_ordering heuristic
        sql="",
        answer_rows=list(gt_rows),
        answer_rows_alts=gt_rows_alts or [],
    )
    result = evaluate_ground_truth(gt_rec, pred_rows, num_decimals=num_decimals)
    return {
        "sr":      float(result.get("sr", 0.0)),
        "row_f1":  float(result.get("row_f1", 0.0)),
        "item_f1": float(result.get("item_f1", 0.0)),
    }


def compute_perf(
    *,
    sr: float,
    row_f1: float,
    item_f1: float,
    weights: dict[str, float],
) -> float:
    """Return the weighted performance score in [0, 1].

    reward = w_sr * sr + w_row * row_f1 + w_item * item_f1
    """
    return float(weights["w_sr"] * sr + weights["w_row"] * row_f1 + weights["w_item"] * item_f1)


# ---------------------------------------------------------------------------
# Public verify_func — plug into EvalConfig.verify_module / verify_func_name
# ---------------------------------------------------------------------------

def verify_func(sample: dict) -> dict:
    """
    Performance verifier for one agent rollout.

    Expected ``sample`` keys (all optional except ``final_answer``):
        final_answer        str               agent's text response (fallback)
        output_csv_path     str | None        path to the CSV the agent wrote;
                                              used instead of final_answer when present
        answer              str | list | None ground-truth (SQL string or pre-parsed rows)
        answer_rows_alts    list | None       alternative valid answer sets for
                                              tie-at-boundary queries; the prediction is
                                              scored against all and the best result kept
        token_usage         dict | None       output of DbAgentRuntime.get_llm_metrics()
        num_decimals        int | None        numeric tolerance for cell comparison
                                              (default 2, matching eval_swan.py)

    Predicted rows are resolved in priority order:
        1. CSV file at ``output_csv_path``  (structured, most reliable)
        2. Free-text parsing of ``final_answer``  (fallback)

    Returns a dict with:
        reward        float   performance score ∈ [0, 1]
                              (cost penalty applied later by experience_updater)
        sr            float
        row_f1        float
        item_f1       float
        llmop_tokens  int     returned separately; folded into reward by the updater
        pred_source   str     "csv" | "text" | "empty"
        reasoning     str     human-readable breakdown
    """
    final_answer = str(sample.get("final_answer") or "").strip()
    ground_truth = sample.get("answer")
    token_usage: dict = sample.get("token_usage") or {}
    num_decimals: int = int(sample.get("num_decimals") or 2)
    # Weights must be injected by the _wrapped_verify closure in
    # enumgrpo._load_verify_func (populated from EvalConfig / config.yaml).
    weights = sample.get("_weights")
    if not weights:
        raise KeyError(
            "verify_func requires '_weights' in the sample dict with keys "
            "'w_sr', 'w_row', 'w_item'. "
            "Make sure it is called through the training loop's wrapped verifier, "
            "or pass EvalConfig reward weights explicitly via sample['_weights']."
        )

    # --- Parse predicted rows: CSV first, free-text fallback ---
    pred_rows: list[list[Any]] = []
    pred_source = "empty"

    output_csv_path = sample.get("output_csv_path")
    if output_csv_path:
        csv_rows = read_csv_rows(output_csv_path)
        if csv_rows is not None:
            pred_rows = csv_rows
            pred_source = "csv"
            logger.debug("verify_func: read %d rows from CSV %s", len(pred_rows), output_csv_path)
        else:
            logger.debug(
                "verify_func: CSV not found at %s; falling back to free-text parsing",
                output_csv_path,
            )

    if pred_source == "empty":
        pred_rows = parse_answer_rows(final_answer)
        if pred_rows:
            pred_source = "text"

    # --- Parse ground-truth rows ---
    gt_rows: list[list[Any]] = []
    if isinstance(ground_truth, list):
        gt_rows = [r if isinstance(r, list) else [r] for r in ground_truth]
    elif isinstance(ground_truth, str) and ground_truth.strip():
        # Try to parse as JSON array-of-arrays (pre-processed datasets).
        parsed = _parse_json_rows(ground_truth.strip())
        if parsed:
            gt_rows = parsed
        else:
            # Raw SQL string or unrecognised format — we can't execute SQL here,
            # so ground truth is unavailable.  Give reward=0.0 rather than the
            # misleading "any non-empty answer = correct" heuristic.
            logger.warning(
                "verify_func: ground truth for sample '%s' is a raw string that cannot "
                "be parsed as rows — setting gt_rows=[] and reward will be 0.0 unless "
                "pred_rows is also empty. Consider pre-parsing answers into row lists.",
                sample.get("id", "?"),
            )
            gt_rows = []

    # Parse alternative answer sets (for tie-at-boundary queries).
    # Mirrors load_ground_truth_jsonl: collects answer_b, answer_c, … in sorted key order,
    # plus any pre-parsed answer_rows_alts field.
    gt_rows_alts: list[list[list[Any]]] = []
    for key in sorted(k for k in sample if k.startswith("answer_")):
        val = sample[key]
        if isinstance(val, list):
            gt_rows_alts.append([r if isinstance(r, list) else [r] for r in val])
    # Also accept a pre-parsed answer_rows_alts field (e.g. from cached rollout dicts).
    for alt in (sample.get("answer_rows_alts") or []):
        if isinstance(alt, list):
            gt_rows_alts.append([r if isinstance(r, list) else [r] for r in alt])

    # --- Score ---
    question = str(sample.get("question") or "")
    scores = score_answer(
        pred_rows, gt_rows,
        gt_rows_alts=gt_rows_alts or None,
        question=question,
        num_decimals=num_decimals,
    )
    sr      = scores["sr"]
    row_f1  = scores["row_f1"]
    item_f1 = scores["item_f1"]

    # --- Cost: LLM-op tokens from the trajectory ---
    # token_usage comes from DbAgentRuntime.get_llm_metrics().
    # We use mcp_total_tokens as the LLM-op cost (llm_map / llm_reduce calls).
    llmop_tokens = int(
        token_usage.get("mcp_total_tokens")
        or (token_usage.get("mcp_prompt_tokens", 0) + token_usage.get("mcp_completion_tokens", 0))
        or 0
    )

    # reward = performance only; cost penalty is applied group-relative in
    # experience_updater so it never dominates performance differences.
    reward = compute_perf(sr=sr, row_f1=row_f1, item_f1=item_f1, weights=weights)

    reasoning = (
        f"source={pred_source}  sr={sr:.3f}  row_f1={row_f1:.3f}  item_f1={item_f1:.3f}  "
        f"llmop_tokens={llmop_tokens}  perf={reward:.4f}  "
        f"(weights: sr={weights['w_sr']} row={weights['w_row']} item={weights['w_item']})"
    )
    logger.info("verify_func: %s", reasoning)

    return {
        "reward":       reward,
        "sr":           sr,
        "row_f1":       row_f1,
        "item_f1":      item_f1,
        "llmop_tokens": llmop_tokens,
        "pred_source":  pred_source,
        "reasoning":    reasoning,
    }
