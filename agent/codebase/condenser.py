"""Custom condenser: prune oversized tool-result observations that have been consumed.

An observation is "consumed" when at least one ActionEvent follows it in the
conversation — meaning the agent has already reasoned about the result and issued
a subsequent tool call.  Keeping the full raw payload after that point only inflates
the context window with data the agent will never re-read.

The pruning strategy:
  - For oversized data-tool results (run_sql, preview_relation, describe_relation,
    list_relations), replace the observation's content with a compact summary note.
  - Leave the most recent N observations unpruned so the agent always has fresh
    context (controlled by keep_recent_obs, default 2).
  - Non-data tools (llm_map, llm_reduce, think, …) are never pruned.

This condenser is designed to run *before* LLMSummarizingCondenser in a
PipelineCondenser so the summariser sees a smaller event list.
"""

from __future__ import annotations

import json
from typing import Any

from openhands.sdk.context.condenser import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event.llm_convertible.action import ActionEvent
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.llm import LLM
from openhands.sdk.llm.message import TextContent


# Tools whose results can balloon with raw row data.
_DATA_TOOLS: frozenset[str] = frozenset(
    {"run_sql", "preview_relation", "describe_relation", "list_relations"}
)

# Default: observations larger than this (in chars) are candidates for pruning.
_DEFAULT_PRUNE_THRESHOLD = 2_000

# How many of the most-recent data-tool observations to leave intact.
_DEFAULT_KEEP_RECENT = 2


class ObservationPruningCondenser(CondenserBase):
    """Prune consumed, oversized tool-result observations from the context view.

    Parameters
    ----------
    prune_threshold:
        Minimum total char length of an observation's text content before it
        is considered for pruning.  Observations smaller than this are always
        kept verbatim.  Default 2000.
    keep_recent_obs:
        Number of most-recent data-tool observations to leave intact regardless
        of size.  Ensures the agent always has its latest results available.
        Default 2.
    """

    def __init__(
        self,
        prune_threshold: int = _DEFAULT_PRUNE_THRESHOLD,
        keep_recent_obs: int = _DEFAULT_KEEP_RECENT,
    ) -> None:
        super().__init__()
        self._prune_threshold = prune_threshold
        self._keep_recent_obs = keep_recent_obs

    def condense(self, view: View, agent_llm: LLM | None = None) -> View:  # noqa: ARG002
        events = list(view.events)
        n = len(events)

        # Walk backwards: track whether any ActionEvent has been seen after each index.
        has_action_after: list[bool] = [False] * n
        seen_action = False
        for i in range(n - 1, -1, -1):
            ev = events[i]
            if isinstance(ev, ActionEvent):
                seen_action = True
            elif isinstance(ev, ObservationEvent) and ev.tool_name in _DATA_TOOLS:
                has_action_after[i] = seen_action

        # Collect indices of consumed data-tool observations that are large enough.
        large_consumed: list[int] = [
            i
            for i in range(n)
            if has_action_after[i] and _obs_text_len(events[i]) > self._prune_threshold
        ]

        # Spare the most recent N from pruning.
        protected: set[int] = set(large_consumed[-self._keep_recent_obs:])
        to_prune: set[int] = set(large_consumed) - protected

        if not to_prune:
            return view

        new_events = [
            _prune_obs(events[i]) if i in to_prune else events[i]
            for i in range(n)
        ]
        return View(
            events=new_events,
            unhandled_condensation_request=view.unhandled_condensation_request,
            condensations=list(view.condensations),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_text_len(ev: Any) -> int:
    """Total chars across all TextContent blocks in an ObservationEvent's content."""
    if not isinstance(ev, ObservationEvent):
        return 0
    obs = ev.observation
    return sum(len(b.text) for b in obs.content if isinstance(b, TextContent))


def _prune_obs(ev: ObservationEvent) -> ObservationEvent:
    """Return a copy of ev with the observation content replaced by a compact summary."""
    obs = ev.observation
    original_len = sum(len(b.text) for b in obs.content if isinstance(b, TextContent))
    summary = _build_summary(obs.content, tool_name=ev.tool_name)
    note = f"[Result pruned from context — {original_len:,} chars → summary] {summary}"
    pruned_obs = obs.model_copy(update={"content": [TextContent(text=note)]})
    return ev.model_copy(update={"observation": pruned_obs})


def _build_summary(content: list[Any], *, tool_name: str) -> str:
    """Build a compact one-line summary from a tool result's content blocks.

    MCPToolObservation prepends a "[Tool 'X' executed.]" preamble block, so we
    try each TextContent block independently for JSON before falling back to the
    full concatenated text.
    """
    texts = [b.text for b in content if isinstance(b, TextContent)]
    if not texts:
        return "(empty result)"

    # Try to parse each block as JSON (skip the preamble block).
    obj = None
    for t in texts:
        try:
            obj = json.loads(t.strip())
            break
        except Exception:
            continue

    raw = " ".join(texts).strip()
    if obj is None:
        return raw[:200] + ("…" if len(raw) > 200 else "")

    if isinstance(obj, dict):
        if "error" in obj:
            return f"error: {str(obj['error'])[:200]}"

        # SQL / relation result: summarise row/column count + first row as sample.
        rows = obj.get("rows")
        cols = obj.get("columns")
        row_count = obj.get("row_count")
        if isinstance(rows, list) and isinstance(cols, list):
            actual = row_count if isinstance(row_count, int) else len(rows)
            col_names = ", ".join(str(c) for c in cols[:10])
            if len(cols) > 10:
                col_names += f", … (+{len(cols) - 10} more)"
            sample = ""
            if rows:
                try:
                    sample = f" | first row: {json.dumps(rows[0], default=str)[:120]}"
                except Exception:
                    pass
            return f"{actual} row(s), cols=[{col_names}]{sample}"

    return raw[:200] + ("…" if len(raw) > 200 else "")
