"""Plan enumerator for structured rollout diversity in training-free GRPO.

Given a natural-language query and a DuckDB path, the PlanEnumerator calls a
cheap LLM to select k diverse and promising axis-value assignments from the
plan-library axis space.  Each assignment is then assembled into a strategy hint
string by concatenating the per-axis snippets defined in plan_library.yaml.

Architecture
------------
  Structure : per-axis snippets in plan_library.yaml  (user-editable)
  Selection : LLM outputs k axis-value assignments as structured JSON
  Assembly  : deterministic concatenation of selected snippets + rationale

Axis space (5 orthogonal dimensions)
-------------------------------------
  D: Execution Paradigm   data_driven | code_driven
  A: Operator Type        map | reduce           (skip when D=code_driven)
  B: Operator Placement   pre_agg | post_agg
  C: Selectivity Scope    full | targeted
  F: Projection Width     narrow | wide          (skip when D=code_driven)

Special value  ``"adaptive"``  selects the Eddies meta-plan (adaptive_hint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import duckdb
import yaml

from utils.llm import chat_complete_async, get_learning_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_VALUES: dict[str, list[str]] = {
    "D": ["data_driven", "code_driven"],
    "A": ["map", "reduce"],
    "B": ["pre_agg", "post_agg"],
    "C": ["full", "targeted"],
    "F": ["narrow", "wide"],
}

# Axes irrelevant when D=code_driven
_CODE_DRIVEN_NA_AXES = {"A", "F"}

# Diversity coverage requirements: each final top-k must include at least one
# assignment from each of these (axis, value) categories.
_DIVERSITY_REQUIREMENTS = [
    ("A", "map"),
    ("A", "reduce"),
    ("D", "code_driven"),
]

# Number of sample rows fetched for the LLM context.
_SAMPLE_ROWS = 20

# Assembly section order (when D != code_driven and != adaptive).
_ASSEMBLY_ORDER = ["D", "A", "B", "C", "F"]


# ---------------------------------------------------------------------------
# Plan library loader
# ---------------------------------------------------------------------------

def load_plan_library(path: str | Path | None = None) -> dict[str, Any]:
    """Load plan_library.yaml.  Falls back to the bundled default when *path* is None."""
    if path is None:
        path = Path(__file__).parent / "plan_library.yaml"
    with Path(path).open("r", encoding="utf-8") as fh:
        lib = yaml.safe_load(fh)
    if not isinstance(lib, dict):
        raise ValueError(f"plan_library.yaml must be a mapping, got {type(lib)}")
    return lib


# ---------------------------------------------------------------------------
# Hint assembly
# ---------------------------------------------------------------------------

def assemble_hint(assignment: dict[str, str], plan_library: dict[str, Any]) -> str:
    """Assemble a strategy hint string from a single axis-value assignment dict.

    The dict must have key ``"D"`` and optionally keys ``"A"``, ``"B"``, ``"C"``,
    ``"F"``, ``"rationale"``.  When ``D == "adaptive"``, the adaptive_hint is
    returned directly.

    Raises ``KeyError`` if a required snippet is not found in the library.
    """
    axes = plan_library.get("axes", {})

    if assignment.get("D") == "adaptive":
        adaptive = (plan_library.get("adaptive_hint") or "").strip()
        return adaptive

    parts: list[str] = []
    is_code_driven = assignment.get("D") == "code_driven"

    for axis in _ASSEMBLY_ORDER:
        if is_code_driven and axis in _CODE_DRIVEN_NA_AXES:
            continue
        value = assignment.get(axis)
        if not value:
            continue
        axis_key = _axis_yaml_key(axis)
        snippet = (axes.get(axis_key) or {}).get(value, "").strip()
        if snippet:
            parts.append(snippet)

    rationale = (assignment.get("rationale") or "").strip()
    if rationale:
        parts.append(f"Rationale: {rationale}")

    return "\n\n".join(parts)


def _axis_yaml_key(axis: str) -> str:
    mapping = {
        "D": "D_execution_paradigm",
        "A": "A_operator_type",
        "B": "B_operator_placement",
        "C": "C_selectivity_scope",
        "F": "F_projection_width",
    }
    return mapping[axis]


# ---------------------------------------------------------------------------
# Diversity post-processing
# ---------------------------------------------------------------------------

def enforce_diversity(
    assignments: list[dict[str, str]],
    k: int,
    *,
    pool: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Ensure the top-k assignments satisfy the diversity requirements.

    Strategy:
    1. Deduplicate assignments by their structural (D, A, B, C, F) key, keeping
       the first occurrence of each distinct combination.
    2. Take the first k from the deduplicated list as the initial selection.
    3. For each unsatisfied (axis, value) requirement, find the highest-ranked
       assignment in the remaining deduplicated pool (or *pool*) that satisfies
       it, and swap it in for the lowest-ranked existing item.

    Raises ``ValueError`` if a diversity requirement cannot be satisfied from
    the provided assignments and pool alone.

    The function never shrinks the list below min(k, len(assignments)).
    """
    candidate_pool: list[dict[str, str]] = list(assignments)
    if pool:
        for p in pool:
            if p not in candidate_pool:
                candidate_pool.append(p)

    # Deduplicate by structural key (D, A, B, C, F), preserving order.
    def _struct_key(a: dict[str, str]) -> tuple:
        return (a.get("D", ""), a.get("A", ""), a.get("B", ""), a.get("C", ""), a.get("F", ""))

    seen: set[tuple] = set()
    deduped: list[dict[str, str]] = []
    for a in candidate_pool:
        key = _struct_key(a)
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    candidate_pool = deduped

    selected: list[dict[str, str]] = list(candidate_pool[:k])

    def _satisfies(a: dict[str, str], axis: str, val: str) -> bool:
        if val == "map" and a.get("D") == "adaptive":
            return False
        if val == "reduce" and a.get("D") == "adaptive":
            return False
        if axis == "A":
            return a.get("A") == val and a.get("D") != "code_driven"
        return a.get(axis) == val

    def _is_covered(axis: str, val: str) -> bool:
        return any(_satisfies(a, axis, val) for a in selected)

    for axis, val in _DIVERSITY_REQUIREMENTS:
        if _is_covered(axis, val):
            continue
        replacement = next(
            (c for c in candidate_pool if c not in selected and _satisfies(c, axis, val)),
            None,
        )
        if replacement is None:
            raise ValueError(
                f"enforce_diversity: cannot satisfy requirement ({axis}={val}) "
                f"from the provided {len(assignments)} assignment(s). "
                f"The LLM must return at least one plan with {axis}={val}."
            )
        if len(selected) < k:
            selected.append(replacement)
        else:
            selected[-1] = replacement

    return selected[:k]


# ---------------------------------------------------------------------------
# DB context helpers
# ---------------------------------------------------------------------------

def _fetch_db_context(db_path: str) -> tuple[str, str]:
    """Return (schema_text, sample_text) for the given DuckDB file.

    Runs in a worker thread (called via asyncio.to_thread).
    Raises on connection or query failure.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # ---- Schema ----
        schema_lines: list[str] = []
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        for (tname,) in tables:
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [tname],
            ).fetchall()
            col_str = ", ".join(f"{c} {t}" for c, t in cols)
            schema_lines.append(f"  {tname}({col_str})")
        schema_text = "\n".join(schema_lines) or "(no tables found)"

        # ---- Sample rows ----
        # Table names from information_schema are controlled by the DB file, not
        # user input, so quoting with double-quotes is sufficient here.
        sample_lines: list[str] = []
        for (tname,) in tables[:3]:   # limit to first 3 tables to keep prompt compact
            safe_tname = tname.replace('"', '""')  # escape embedded quotes
            try:
                res = con.execute(
                    f'SELECT * FROM "{safe_tname}" LIMIT {_SAMPLE_ROWS}'
                )
                cols = [d[0] for d in res.description]
                rows = res.fetchall()
                sample_lines.append(f"-- {tname} (first {len(rows)} rows)")
                sample_lines.append(", ".join(cols))
                for row in rows:
                    sample_lines.append(", ".join("NULL" if v is None else str(v)[:60] for v in row))
                sample_lines.append("")
            except Exception as exc:
                sample_lines.append(f"-- {tname}: (could not sample: {exc})")
        sample_text = "\n".join(sample_lines).strip() or "(no sample available)"
        return schema_text, sample_text
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a database query planning assistant.  Your task is to select a diverse
set of k execution strategies for a database agent that answers analytical queries
over dirty or incomplete DuckDB databases.

The agent has two LLM operators available:
  - llm_map    : applies the LLM to individual rows (or small batches) one at a time
  - llm_reduce : receives the entire pre-filtered result set in one prompt

The agent can also ask the LLM to generate a deterministic SQL rule (CASE WHEN,
regexp_replace, etc.) that DuckDB executes natively — paying tokens only once
for rule generation.

You will be given:
  1. A natural-language query
  2. The database schema
  3. A sample of raw data rows (so you can observe actual dirtiness)
  4. Descriptions of each axis option

Output ONLY a JSON array of exactly k objects with no other text.  Each object:
  {
    "D": "data_driven" | "code_driven" | "adaptive",
    "A": "map" | "reduce",          // omit when D is code_driven or adaptive
    "B": "pre_agg" | "post_agg",    // omit when D is adaptive
    "C": "full" | "targeted",       // omit when D is adaptive
    "F": "narrow" | "wide",         // omit when D is code_driven or adaptive
    "rationale": "<one sentence>"
  }
Exactly one object MUST have D="code_driven", at least one MUST have A="map",
and at least one MUST have A="reduce".
No two objects may share the same combination — every assignment
in the array must be structurally distinct.
"""


def _build_user_prompt(
    question: str,
    schema_text: str,
    sample_text: str,
    k: int,
    axes_descriptions: str,
) -> str:
    return f"""\
## Query
{question}

## Database schema
{schema_text}

## Sample data
{sample_text}

## Axis option descriptions
{axes_descriptions}

## Instructions
Select exactly {k} axis-value assignments.  Favour combinations that are most
promising for this specific query and data.  Ensure the diversity requirements:
at least one A=map, one A=reduce, one D=code_driven.  All {k} assignments must
be structurally distinct — no two may share the same (D, A, B, C, F) values.
"""


def _build_axes_descriptions(plan_library: dict[str, Any]) -> str:
    """Format the per-axis option descriptions from plan_library for the prompt."""
    axes = plan_library.get("axes", {})
    lines: list[str] = []
    order = [
        ("D_execution_paradigm",   "D: Execution Paradigm"),
        ("A_operator_type",        "A: Operator Type (skip when D=code_driven)"),
        ("B_operator_placement",   "B: Operator Placement"),
        ("C_selectivity_scope",    "C: Selectivity Scope"),
        ("F_projection_width",     "F: Projection Width (skip when D=code_driven)"),
    ]
    for yaml_key, label in order:
        options = axes.get(yaml_key) or {}
        lines.append(f"### {label}")
        for val, desc in options.items():
            lines.append(f"  {val}: {str(desc).strip()}")
        lines.append("")
    lines.append("### Adaptive meta-plan (D=adaptive)")
    adaptive = (plan_library.get("adaptive_hint") or "").strip()
    lines.append(f"  adaptive: {adaptive}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def _parse_assignments(raw: str, k: int) -> list[dict[str, str]]:
    """Extract and validate axis assignments from the LLM response string."""
    raw = raw.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Find the first JSON array in the response.
    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if arr_match:
        raw = arr_match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}\nRaw: {raw[:500]}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    validated: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        a: dict[str, str] = {}

        d_val = str(item.get("D", "")).strip()
        if d_val not in ("data_driven", "code_driven", "adaptive"):
            continue
        a["D"] = d_val

        if d_val == "adaptive":
            a["rationale"] = str(item.get("rationale", "")).strip()
            validated.append(a)
            continue

        if d_val == "code_driven":
            for key in ("B", "C"):
                v = str(item.get(key, "")).strip()
                if v in _VALID_VALUES[key]:
                    a[key] = v
            a["rationale"] = str(item.get("rationale", "")).strip()
            validated.append(a)
            continue

        # data_driven
        for key in ("A", "B", "C", "F"):
            v = str(item.get(key, "")).strip()
            if v in _VALID_VALUES[key]:
                a[key] = v
        a["rationale"] = str(item.get("rationale", "")).strip()
        validated.append(a)

    return validated


# ---------------------------------------------------------------------------
# PlanEnumerator
# ---------------------------------------------------------------------------

class PlanEnumerator:
    """LLM-based plan enumerator for structured rollout diversity.

    Usage::

        library = load_plan_library()
        enumerator = PlanEnumerator(library)

        # async context:
        hints = await enumerator.select(question, db_path, k=5)
        # hints is a list[str] of assembled strategy hint strings

        # sync context:
        hints = enumerator.select_sync(question, db_path, k=5)
    """

    def __init__(
        self,
        plan_library: dict[str, Any],
        *,
        model: str | None = None,
    ) -> None:
        self.plan_library = plan_library
        self._axes_descriptions = _build_axes_descriptions(plan_library)
        self.model = model or get_learning_model()

    async def select(
        self,
        question: str,
        db_path: str,
        k: int,
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Return (hints, assignments) -- k assembled hint strings and their raw axis dicts."""
        assignments = await self._enumerate(question, db_path, k)
        hints = [assemble_hint(a, self.plan_library) for a in assignments]
        return hints, assignments

    def select_sync(
        self,
        question: str,
        db_path: str,
        k: int,
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Synchronous wrapper around :meth:`select`."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self.select(question, db_path, k), loop
            )
            return fut.result()
        return loop.run_until_complete(self.select(question, db_path, k))

    async def _enumerate(
        self,
        question: str,
        db_path: str,
        k: int,
    ) -> list[dict[str, str]]:
        """Core enumeration: fetch context, call LLM, post-process.

        The reference plans are passed as the fallback *pool* so that
        ``enforce_diversity`` can pull in missing diversity coverage from the
        canonical set rather than raising.  This preserves all valid LLM
        assignments while only supplementing the slots that need it.
        """
        schema_text, sample_text = await asyncio.to_thread(
            _fetch_db_context, db_path
        )
        raw_assignments = await self._call_llm(question, schema_text, sample_text, k)
        return enforce_diversity(raw_assignments, k, pool=list(_REFERENCE_PLANS))

    async def _call_llm(
        self,
        question: str,
        schema_text: str,
        sample_text: str,
        k: int,
    ) -> list[dict[str, str]]:
        user_prompt = _build_user_prompt(
            question=question,
            schema_text=schema_text,
            sample_text=sample_text,
            k=k,
            axes_descriptions=self._axes_descriptions,
        )
        raw_response = await chat_complete_async(
            self.model,
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.3,
        )
        assignments = _parse_assignments(raw_response, k)
        logger.debug(
            "PlanEnumerator: LLM returned %d valid assignments for question '%.60s'",
            len(assignments), question,
        )
        return assignments


# ---------------------------------------------------------------------------
# Rule-based random enumerator (drop-in replacement for PlanEnumerator)
# ---------------------------------------------------------------------------

# The seven canonical reference plans that span the axis space.
_REFERENCE_PLANS: list[dict[str, str]] = [
    {"D": "data_driven", "A": "map",    "B": "post_agg", "C": "full",     "F": "wide"},
    {"D": "data_driven", "A": "map",    "B": "post_agg", "C": "full",     "F": "narrow"},
    {"D": "data_driven", "A": "map",    "B": "pre_agg",  "C": "targeted", "F": "narrow"},
    {"D": "data_driven", "A": "reduce", "B": "post_agg", "C": "full",     "F": "wide"},
    {"D": "data_driven", "A": "reduce", "B": "pre_agg",  "C": "targeted", "F": "narrow"},
    {"D": "code_driven",                "B": "post_agg", "C": "full"},
    {"D": "adaptive"},
]


class RandomPlanEnumerator:
    """Rule-based drop-in replacement for :class:`PlanEnumerator`.

    Instead of calling an LLM, it randomly samples k distinct plans from the
    seven canonical reference plans, then passes them through
    :func:`enforce_diversity` to guarantee structural coverage.

    The constructor signature and public API are identical to
    :class:`PlanEnumerator` so the two classes are interchangeable::

        library = load_plan_library()
        enumerator = RandomPlanEnumerator(library)           # no model needed
        hints, axes = await enumerator.select(question, db_path, k=5)
        hints, axes = enumerator.select_sync(question, db_path, k=5)

    The ``model`` parameter is accepted but ignored.
    ``question`` and ``db_path`` are accepted but not used (no LLM / DB call).
    """

    def __init__(
        self,
        plan_library: dict[str, Any],
        *,
        model: str | None = None,  # accepted for interface compatibility; unused
    ) -> None:
        self.plan_library = plan_library
        self.model = model  # kept for attribute parity; never used

    async def select(
        self,
        question: str,
        db_path: str,
        k: int,
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Return (hints, assignments) sampled from the reference plan set."""
        assignments = self._sample(k)
        hints = [assemble_hint(a, self.plan_library) for a in assignments]
        return hints, assignments

    def select_sync(
        self,
        question: str,
        db_path: str,
        k: int,
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Synchronous wrapper -- no event loop required."""
        assignments = self._sample(k)
        hints = [assemble_hint(a, self.plan_library) for a in assignments]
        return hints, assignments

    def _sample(self, k: int) -> list[dict[str, str]]:
        """Randomly sample k distinct plans, then enforce diversity requirements."""
        shuffled = random.sample(_REFERENCE_PLANS, min(k, len(_REFERENCE_PLANS)))
        # If k exceeds the reference set size, cycle through shuffled copies.
        while len(shuffled) < k:
            extra = random.sample(_REFERENCE_PLANS, min(k - len(shuffled), len(_REFERENCE_PLANS)))
            shuffled.extend(extra)
        return enforce_diversity(shuffled, k)
