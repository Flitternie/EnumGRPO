"""Experience updater for training-free GRPO adapted to the DB agent.

This module mirrors the multi-stage pipeline of `utu.practice.experience_updater`:

1) Single-rollout summaries
2) Group-relative advantages per query
3) Per-query experience update operations
4) Batch-level consolidation of all operations
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import PracticeConfig
from .rollout_manager import RolloutResult
from .utils import TaskRecorder, distill_bar as _distill_bar, tqdm_write as _tqdm_write
from utils.llm import chat_complete_async, get_learning_model

logger = logging.getLogger(__name__)



def _load_prompts() -> dict[str, str]:
    p = Path(__file__).resolve().parent / "prompts" / "experience.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Distillation log helpers
# ---------------------------------------------------------------------------

def _fmt_exec_profile(exec_profile: list[dict] | None) -> str:
    """Produce a compact one-line-per-step summary of the exec profile."""
    if not exec_profile:
        return "(no profile)"
    parts: list[str] = []
    for i, step in enumerate(exec_profile, 1):
        tool = step.get("tool", "?")
        ms = step.get("duration_ms")
        op = step.get("op_preview")
        inp = step.get("input_rows")
        out = step.get("row_count")
        tok_in = step.get("llm_input_tokens")
        tok_out = step.get("llm_output_tokens")
        err = step.get("error")

        seg = f"{i}.{tool}"
        if ms is not None:
            seg += f" {ms}ms"
        if op:
            seg += f" [{op[:60]}]" if len(op) > 60 else f" [{op}]"
        if inp is not None and out is not None:
            seg += f" {inp}→{out}r"
        elif out is not None:
            seg += f" {out}r"
        if tok_in is not None or tok_out is not None:
            seg += f" tok:{tok_in or 0}/{tok_out or 0}"
        if err:
            seg += " ERR"
        parts.append(seg)
    return " | ".join(parts)


def _write_distillation_log(
    log_dir: "Path",
    *,
    step: int,
    epoch: int,
    batch: int,
    critiques: list[dict],
    batch_revision_plan: list[dict],
    experiences_before: dict[str, str],
    experiences_after: dict[str, str],
) -> None:
    """Write a structured distillation log for one training step.

    Output: <log_dir>/distillation/step_NNNN.json

    Top-level structure::

        {
          "step": N, "epoch": N, "batch": N,
          "experiences_before": {...},
          "experiences_after": {...},
          "batch_decisions": [...],   // final consolidation ops
          "queries": [
            {
              "question": "...",
              "ground_truth": "...",
              "rollouts": [
                {
                  "sample_id": "...",
                  "plan": "D=data_driven A=map ...",
                  "exec_profile": "<compact one-liner>",
                  "perf_reward": 0.8,
                  "llmop_tokens": 1200,
                  "reward": 0.75,
                  "advantage": +1.2,
                  "perf_adv": +0.9,
                  "eff_adv": -0.3,
                  "trajectory_summary": "..."
                }, ...
              ],
              "group_analysis": "...",      // full LLM group-advantage response
              "proposed_experiences": "...", // <Experiences> block extracted
              "group_decisions": [...]       // per-group ADD/UPDATE/DELETE ops
            }, ...
          ]
        }
    """
    from pathlib import Path as _Path
    out_dir = _Path(log_dir) / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{step:04d}.json"

    def _plan_label(axes: dict | None) -> str:
        if not axes:
            return ""
        d = axes.get("D", "")
        if d == "adaptive":
            return "adaptive"
        parts = [f"{k}={axes[k]}" for k in ("D", "A", "B", "C", "F") if axes.get(k)]
        return " ".join(parts)

    queries: list[dict] = []
    for critique in critiques:
        rollout_entries = critique.get("rollouts") or []
        if not rollout_entries:
            continue
        question = rollout_entries[0].get("raw_question", "")
        rollouts_out = []
        for r in rollout_entries:
            rollouts_out.append({
                "sample_id": r.get("sample_id"),
                "plan": _plan_label(r.get("plan_axes")),
                "exec_profile": r.get("exec_profile_summary", ""),
                "perf_reward": r.get("perf_reward"),
                "llmop_tokens": r.get("llmop_tokens"),
                "num_steps": r.get("num_steps"),
                "reward": round(r.get("reward", 0.0), 4),
                "advantage": round(r.get("advantage", 0.0), 4),
                "perf_adv": round(r.get("perf_adv", 0.0), 4),
                "eff_adv": round(r.get("eff_adv", 0.0), 4),
                "trajectory_summary": r.get("trajectory_summary", ""),
            })
        queries.append({
            "question": question,
            "ground_truth": rollout_entries[0].get("correct_answer"),
            "rollouts": rollouts_out,
            "group_analysis": critique.get("critique", ""),
            "proposed_experiences": critique.get("experiences", ""),
            "group_decisions": critique.get("operations") or [],
        })

    doc = {
        "step": step,
        "epoch": epoch,
        "batch": batch,
        "experiences_before": experiences_before,
        "experiences_after": experiences_after,
        "batch_decisions": batch_revision_plan,
        "queries": queries,
    }
    try:
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write distillation log to %s", out_path)


@dataclass
class ExperienceUpdater:
    """Full GRPO-style experience updater."""

    cfg: PracticeConfig
    agent_objective: str
    learning_objective: str
    model: str

    def __init__(self, cfg: PracticeConfig) -> None:
        self.cfg = cfg
        self.agent_objective = cfg.practice.agent_objective or "A DB agent that answers questions using DuckDB."
        self.learning_objective = (
            cfg.practice.learning_objective
            or "Learn stable, reusable strategies for accurate, safe, and efficient DB usage."
        )
        self.model = get_learning_model()
        self.prompts = _load_prompts()
        self._distillation_concurrency: int = cfg.practice.distillation_concurrency

    async def run(
        self,
        rollouts: list[RolloutResult],
        recorder: TaskRecorder,
        *,
        log_dir: "Path | None" = None,
        step: int = 0,
        epoch: int = 0,
        batch: int = 0,
    ) -> dict[str, str]:
        """Top-level entry: derive a new experience set from recent rollouts."""
        experiences_before = dict(recorder.experiences or {})

        # 1) Summarize trajectory for each rollout (per problem).
        problem_to_summarized = await self._single_rollout_summary(
            rollouts=rollouts,
            given_ground_truth=self.cfg.practice.given_ground_truth,
        )

        # 2) Generate semantic group advantages based on summarized rollouts.
        per_problem_exps = await self._group_advantage(
            problem_to_summarized,
            given_ground_truth=self.cfg.practice.given_ground_truth,
            num_experiences=self.cfg.practice.num_experiences_per_query,
        )

        # 3) Per-group experience updates (operations).
        critiques = await self._group_update(
            recorder=recorder,
            new_experiences=per_problem_exps,
        )

        # 4) Batch-level consolidation of all operations.
        new_experiences, batch_revision_plan = await self._batch_update(
            recorder=recorder,
            critiques=critiques,
        )

        # 5) Assign new experience IDs: new ADDs from _batch_update already use
        # G-prefixed keys; existing IDs are preserved, so the pool stays stable
        # across steps (no positional renumbering that would break UPDATE/DELETE
        # references in future steps).
        recorder.experiences_update(new_experiences)

        # 6) Write structured distillation log to run log dir.
        if log_dir is not None:
            _write_distillation_log(
                log_dir=log_dir,
                step=step, epoch=epoch, batch=batch,
                critiques=critiques,
                batch_revision_plan=batch_revision_plan,
                experiences_before=experiences_before,
                experiences_after=new_experiences,
            )

        return new_experiences

    def _group_rollouts_by_question(
        self,
        rollouts: Iterable[RolloutResult],
    ) -> dict[str, list[RolloutResult]]:
        grouped: dict[str, list[RolloutResult]] = defaultdict(list)
        for r in rollouts:
            grouped[r.sample.question].append(r)
        return grouped

    async def _single_rollout_summary(
        self,
        rollouts: list[RolloutResult],
        *,
        given_ground_truth: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        """Summarize each rollout's trajectory, grouped by question.

        For each query group the following are computed before summarisation:

        1. Within-group composite cost normalisation:
               token_rank = (tokens - min_tokens) / max(max_tokens - min_tokens, 1)
               step_rank  = (steps  - min_steps)  / max(max_steps  - min_steps,  1)
               cost_rank  = w_token·token_rank + w_step·step_rank
               final_reward = perf - w_cost * cost_rank

        2. Group-relative advantage (standard GRPO):
               advantage = (final_reward - mean(final_reward_in_group))
                           / max(std(final_reward_in_group), 1e-8)

           Dividing by std makes the advantage scale-invariant across groups —
           a "+1.0" advantage consistently means "1 std above average for this
           query" regardless of how spread the group's rewards are.  When all
           rollouts score identically (std≈0), advantages collapse to ~0,
           correctly signalling "nothing to learn here".

        The advantage is what gets passed to the experience-distillation LLM so
        it sees relative signal (+good / -bad) rather than absolute scores that
        are incomparable across queries of different difficulty.

        All groups are included for distillation regardless of whether they all
        failed or all succeeded: the LLM can learn from consistent failure
        patterns and from what reliably works.  Only groups where all rollouts
        score identically (variance < 1e-8 after cost adjustment) are dropped,
        since advantages collapse to 0 and provide truly no relative signal.
        """
        problems_to_rollouts = self._group_rollouts_by_question(rollouts)
        w_cost: float = self.cfg.evaluation.reward_w_cost
        w_token: float = self.cfg.evaluation.reward_w_token
        w_steps: float = self.cfg.evaluation.reward_w_steps

        # List of per-rollout score tuples — computed per group.
        # Fields: (rollout, final_reward, combined_advantage, perf_adv, eff_adv)
        all_rollouts_to_process: list[tuple[RolloutResult, float, float, float, float]] = []
        for rs in problems_to_rollouts.values():
            perfs = [float(each.reward or 0.0) for each in rs]
            if not perfs:
                continue

            # --- Within-group composite cost normalisation ---
            # cost_rank = w_token·token_rank + w_step·step_rank
            # Both sub-ranks are min-max normalised within the group so the
            # penalty is always relative, never absolute.

            tokens = [each.llmop_tokens for each in rs]
            min_t, max_t = min(tokens), max(tokens)
            token_range = max(max_t - min_t, 1)
            token_ranks = [(r.llmop_tokens - min_t) / token_range for r in rs]

            steps = [each.num_steps for each in rs]
            min_s, max_s = min(steps), max(steps)
            step_range = max(max_s - min_s, 1)
            step_ranks = [(r.num_steps - min_s) / step_range for r in rs]

            cost_ranks = [w_token * tr + w_steps * sr for tr, sr in zip(token_ranks, step_ranks)]
            final_rewards = [p - w_cost * c for p, c in zip(perfs, cost_ranks)]

            def _z_scores(values: list[float]) -> list[float]:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    var = sum((v - mean) ** 2 for v in values) / len(values)
                    std = max(math.sqrt(var), 1e-8)
                else:
                    std = 1e-8  # single-rollout group — z-scores collapse to ~0
                return [(v - mean) / std for v in values]

            combined_advs = _z_scores(final_rewards)
            perf_advs = _z_scores(perfs)
            # Negate composite cost_ranks so that fewer tokens/steps → positive efficiency advantage.
            eff_advs = _z_scores([-c for c in cost_ranks])

            for r, fr, adv, padv, eadv in zip(rs, final_rewards, combined_advs, perf_advs, eff_advs):
                all_rollouts_to_process.append((r, fr, adv, padv, eadv))

        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sem = asyncio.Semaphore(self._distillation_concurrency)

        async def summarize_one(r: RolloutResult, final_reward: float, advantage: float, perf_adv: float, eff_adv: float) -> None:
            async with sem:
                try:
                    sp = (
                        self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP"]
                        .replace("{{ agent_objective }}", self.agent_objective)
                        .replace("{{ learning_objective }}", self.learning_objective)
                    )
                    traj_json = json.dumps(r.trajectory or [], ensure_ascii=False, indent=2)

                    # Build a compact execution profile to accompany the trajectory.
                    profile_lines: list[str] = []
                    if r.exec_profile:
                        profile_lines.append("Execution Profile (one entry per MCP tool call):")
                        for i, step in enumerate(r.exec_profile, 1):
                            parts = [f"{i}. {step['tool']}"]
                            if step.get("duration_ms") is not None:
                                parts.append(f"{step['duration_ms']}ms")
                            # Operation context: SQL snippet, relation name, or question.
                            if step.get("op_preview"):
                                parts.append(f"op: {step['op_preview']}")
                            # Cardinality: input→output for LLM ops, output-only for SQL ops.
                            inp_rows = step.get("input_rows")
                            out_rows = step.get("row_count")
                            if inp_rows is not None and out_rows is not None:
                                parts.append(f"in: {inp_rows} → out: {out_rows} rows")
                            elif inp_rows is not None:
                                parts.append(f"in: {inp_rows} rows")
                            elif out_rows is not None:
                                parts.append(f"out: {out_rows} rows")
                            # LLM token breakdown (LLM ops only).
                            llm_in = step.get("llm_input_tokens")
                            llm_out = step.get("llm_output_tokens")
                            if llm_in is not None or llm_out is not None:
                                parts.append(f"tokens: {llm_in or 0} in / {llm_out or 0} out")
                            if step.get("error"):
                                parts.append(f"[ERROR: {step['error']}]")
                            line = "  " + " | ".join(parts)
                            # Result preview: cols and first 2 rows, inline.
                            rp = step.get("result_preview")
                            if rp and rp.get("cols"):
                                cols_str = ", ".join(rp["cols"])
                                rows_str = "  ".join(
                                    "(" + ", ".join(f'"{v}"' for v in row) + ")"
                                    for row in rp.get("rows") or []
                                )
                                line += f"\n    cols: [{cols_str}]"
                                if rows_str:
                                    line += f"  sample: {rows_str}"
                            profile_lines.append(line)
                    profile_text = "\n".join(profile_lines) if profile_lines else "(no execution profile available)"

                    # Read first 5 rows of the agent's output CSV so the distiller
                    # can see what the agent actually produced.
                    agent_output_text = "(not available)"
                    if r.output_csv_path:
                        try:
                            with open(r.output_csv_path, newline="", encoding="utf-8") as fh:
                                reader = csv.reader(fh)
                                csv_rows = [row for _, row in zip(range(6), reader)]
                            if csv_rows:
                                header = csv_rows[0]
                                data = csv_rows[1:]
                                lines = [", ".join(header)]
                                for row in data:
                                    lines.append(", ".join(row))
                                agent_output_text = "\n".join(lines)
                        except Exception:
                            logger.warning(
                                "Could not read agent output CSV for sample %s at %s; "
                                "distillation will proceed without it.",
                                r.sample.id, r.output_csv_path,
                            )

                    up = (
                        self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP"]
                        .replace("{{ question }}", r.sample.question)
                        .replace("{{ trajectory }}", f"{profile_text}\n\nTrajectory:\n{traj_json}")
                        .replace(
                            "{{ answer }}",
                            json.dumps(r.sample.answer) if (given_ground_truth and r.sample.answer is not None) else "[REDACTED]",
                        )
                        .replace("{{ critique }}", r.reasoning or "[No critique provided]")
                        .replace("{{ agent_output }}", agent_output_text)
                        .replace("{{ perf_reward }}", f"{float(r.reward or 0.0):.4f}")
                        .replace("{{ advantage }}", f"{advantage:+.3f}")
                    )
                    response = await chat_complete_async(
                        self.model, system=sp, user=up, temperature=0.1
                    )
                    results[r.sample.question].append(
                        {
                            "sample_id": r.sample.id,
                            "raw_question": r.sample.question,
                            "correct_answer": r.sample.answer,
                            "plan_axes": r.plan_axes,
                            "plan_hint": r.plan_hint,
                            "exec_profile_summary": _fmt_exec_profile(r.exec_profile),
                            "perf_reward": float(r.reward or 0.0),
                            "llmop_tokens": r.llmop_tokens,
                            "num_steps": r.num_steps,
                            "reward": final_reward,
                            "advantage": advantage,
                            "perf_adv": perf_adv,
                            "eff_adv": eff_adv,
                            "trajectory_summary": response,
                        }
                    )
                except Exception:
                    logger.exception(
                        "Single-rollout summary failed for sample %s", r.sample.id
                    )
                    return

        tasks = [asyncio.create_task(summarize_one(r, fr, adv, padv, eadv)) for r, fr, adv, padv, eadv in all_rollouts_to_process]
        if tasks:
            bar = _distill_bar(len(tasks), "Summarise rollouts")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results

    async def _group_advantage(
        self,
        problem_to_summarized_rollouts: dict[str, list[dict]],
        *,
        given_ground_truth: bool,
        num_experiences: int,
    ) -> list[dict]:
        """Generate per-problem experiences based on summarized rollouts."""
        all_rollouts: list[list[dict]] = []
        for rs in problem_to_summarized_rollouts.values():
            scores = [float(each["reward"]) for each in rs]
            if not scores:
                continue
            # Skip only when all rollouts scored identically AND have identical
            # costs: advantages are all exactly 0, so the LLM has no relative
            # signal at all.  All-fail and all-succeed groups are kept because
            # the distiller can learn from consistent failure patterns and from
            # what reliably works (including efficiency differences).
            if len(scores) > 1:
                avg_score = sum(scores) / len(scores)
                var = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                if var < 1e-8:
                    continue
            all_rollouts.append(rs)

        sem = asyncio.Semaphore(self._distillation_concurrency)
        results: list[dict] = []

        async def per_problem(rollouts_per_problem: list[dict]) -> None:
            async with sem:
                try:
                    # Show group-relative advantage so the LLM sees +/- signal
                    # rather than absolute scores that differ across queries.
                    def _fmt_advantage(entry: dict) -> str:
                        if given_ground_truth:
                            return (
                                f"Advantage {entry['advantage']:+.3f} "
                                f"| Correctness {entry['perf_adv']:+.3f}, "
                                f"Efficiency {entry['eff_adv']:+.3f}"
                            )
                        return "[REDACTED]"

                    def _fmt_plan_label(entry: dict) -> str:
                        axes = entry.get("plan_axes")
                        if not axes:
                            return ""
                        d = axes.get("D", "")
                        if d == "adaptive":
                            return " | plan: adaptive"
                        if d == "code_driven":
                            b = axes.get("B", "")
                            c = axes.get("C", "")
                            label = "D=code_driven"
                            if b:
                                label += f", B={b}"
                            if c:
                                label += f", C={c}"
                            return f" | plan: {label}"
                        parts = []
                        for key in ("D", "A", "B", "C", "F"):
                            v = axes.get(key)
                            if v:
                                parts.append(f"{key}={v}")
                        return " | plan: " + ", ".join(parts) if parts else ""

                    formatted = "\n\n".join(
                        f"Attempt {i + 1} ({_fmt_advantage(each)}{_fmt_plan_label(each)}):\n"
                        f"{each['trajectory_summary']}"
                        for i, each in enumerate(rollouts_per_problem)
                    )
                    sp = (
                        self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_SP"]
                        .replace("{{ agent_objective }}", self.agent_objective)
                        .replace("{{ learning_objective }}", self.learning_objective)
                        .replace("{{ num_experiences }}", str(num_experiences))
                    )
                    up = (
                        self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_UP"]
                        .replace("{{ question }}", rollouts_per_problem[0]["raw_question"])
                        .replace(
                            "{{ answer }}",
                            json.dumps(rollouts_per_problem[0]["correct_answer"]) if given_ground_truth else "[REDACTED]",
                        )
                        .replace("{{ trajectories }}", formatted)
                    )
                    response = await chat_complete_async(
                        self.model, system=sp, user=up, temperature=0.2
                    )
                    pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>", re.DOTALL | re.IGNORECASE)
                    match = pattern.search(response)
                    experiences = match.group(1).strip() if match else ""
                    results.append(
                        {
                            "rollouts": rollouts_per_problem,
                            "critique": response,
                            "experiences": experiences,
                        }
                    )
                except Exception:
                    logger.exception(
                        "Group advantage failed for question: %.80s...",
                        rollouts_per_problem[0].get("raw_question", ""),
                    )
                    return

        tasks = [asyncio.create_task(per_problem(rs)) for rs in all_rollouts]
        if tasks:
            bar = _distill_bar(len(tasks), "Group advantage  ")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results

    async def _group_update(
        self,
        recorder: TaskRecorder,
        new_experiences: list[dict],
    ) -> list[dict]:
        """Group-level update operations for each problem's experiences."""
        sem = asyncio.Semaphore(self._distillation_concurrency)
        results: list[dict] = []

        async def per_group(new_exp: dict) -> None:
            async with sem:
                try:
                    curr = recorder.experiences or {}
                    formatted_experiences = (
                        "\n".join(f"[{i}]. {e}" for i, e in curr.items()) if curr else "None"
                    )
                    sp = (
                        self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP"]
                        .replace("{{ agent_objective }}", self.agent_objective)
                        .replace("{{ learning_objective }}", self.learning_objective)
                    )
                    up = (
                        self.prompts["GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP"]
                        .replace("{{ existing_experiences }}", formatted_experiences)
                        .replace("{{ new_experiences }}", new_exp["experiences"])
                    )
                    operations: list[dict] = []
                    parse_error: str | None = None
                    for attempt in range(2):
                        response = await chat_complete_async(
                            self.model, system=sp, user=up, temperature=0.2
                        )
                        payload = response.split("```json")[-1].split("```")[0]
                        try:
                            operations = json.loads(payload)
                            parse_error = None
                        except json.JSONDecodeError as exc:
                            parse_error = str(exc)
                            logger.warning(
                                "Group update returned invalid JSON for '%.60s' (attempt %d/2): %s",
                                new_exp.get("experiences", "")[:60], attempt + 1, exc,
                            )
                            if attempt == 0:
                                up = up + (
                                    "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
                                    "Return ONLY a valid JSON array inside a ```json ... ``` block."
                                )
                            continue
                        # Validate: at least one non-NONE operation required.
                        if any(op.get("operation", "NONE") != "NONE" for op in operations):
                            break
                        if attempt == 0:
                            # Retry with an explicit reminder that quotes the input back.
                            up = up + (
                                "\n\nIMPORTANT REMINDER: Your previous response had all operations "
                                "set to NONE. You MUST produce at least one ADD, UPDATE, or DELETE.\n"
                                "The new experiences you were given are:\n"
                                f"{new_exp['experiences']}\n\n"
                                "Pick the most valuable insight above and apply ADD (if it is genuinely "
                                "new), UPDATE (if it refines an existing entry), or DELETE (if it "
                                "contradicts one). Explain your choice in the JSON content field."
                            )
                            logger.warning(
                                "Group update returned all-NONE operations for '%.60s'; retrying.",
                                new_exp.get("experiences", "")[:60],
                            )
                    if parse_error:
                        logger.warning(
                            "Group update still returned invalid JSON after 2 attempts for '%.60s'; "
                            "dropping this group from distillation.",
                            new_exp.get("experiences", "")[:60],
                        )
                        return
                    results.append({"operations": operations, **new_exp})
                except Exception:
                    logger.exception(
                        "Group update failed for experience: %.80s...",
                        new_exp.get("experiences", ""),
                    )
                    return

        tasks = [asyncio.create_task(per_group(ne)) for ne in new_experiences]
        if tasks:
            bar = _distill_bar(len(tasks), "Group update     ")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results

    async def _batch_update(
        self,
        recorder: TaskRecorder,
        critiques: list[dict],
        max_retries: int = 3,
    ) -> tuple[dict[str, str], list[dict]]:
        """Batch-level consolidation of all operations into a new experience pool.

        Returns (new_experiences_dict, revision_plan) where revision_plan is the
        raw list of operation dicts from the LLM, used for distillation logging.
        """
        curr_exps = recorder.experiences or {}

        all_operations: list[dict] = []
        for each in critiques:
            all_operations.extend(each.get("operations", []))

        def _format_exp_and_ops(experiences: dict[str, str], operations: list[dict]) -> tuple[str, str]:
            """Return (exp_and_update_ops_text, pending_add_ops_text).

            ADD operations (id=null) are separated so the batch LLM sees them
            alongside the full existing pool and can detect semantic duplicates.
            """
            if not operations:
                return "No batch operations.", "None."
            lines: list[str] = []
            for id_, exp in experiences.items():
                curr = [f"Experience {id_}:", f"Content: {exp}"]
                related = [op for op in operations if op.get("id") == id_]
                if related:
                    curr.append("Related Operations:")
                    curr.extend(json.dumps(op, ensure_ascii=False, indent=2) for op in related)
                else:
                    curr.append("No related operations.")
                lines.append("\n".join(curr))
            exp_text = "\n\n".join(lines) if lines else "None."

            add_ops = [op for op in operations if not op.get("id") and op.get("operation") == "ADD"]
            add_text = "\n".join(json.dumps(op, ensure_ascii=False, indent=2) for op in add_ops) if add_ops else "None."
            return exp_text, add_text

        experiences = curr_exps
        revision_plan: list[dict] = []

        for attempt in range(max_retries):
            try:
                _tqdm_write(
                    f"    Batch consolidate  ({len(all_operations)} ops, attempt {attempt + 1}/{max_retries})…"
                )
                sp = (
                    self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP"]
                    .replace("{{ agent_objective }}", self.agent_objective)
                    .replace("{{ learning_objective }}", self.learning_objective)
                )
                exp_text, add_text = _format_exp_and_ops(experiences, all_operations)
                up = (
                    self.prompts["BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP"]
                    .replace("{{ experiences_and_operations }}", exp_text)
                    .replace("{{ pending_add_operations }}", add_text)
                )
                response = await chat_complete_async(
                    self.model, system=sp, user=up, temperature=0.2
                )
                payload = response.split("```json")[-1].split("```")[0]
                revision_plan = json.loads(payload)
                break
            except Exception:
                logger.exception(
                    "Batch update failed (attempt %d/%d); will %s.",
                    attempt + 1,
                    max_retries,
                    "retry" if attempt + 1 < max_retries else "skip and keep existing experiences",
                )

        max_id = len(experiences)
        new_exps = copy.deepcopy(experiences)
        for plan in revision_plan:
            operation = plan.get("operation", "ADD")
            content = plan.get("content", "")
            target_id = plan.get("id", None)
            if not content:
                continue
            if operation == "ADD":
                new_exps[f"G{max_id}"] = content
                max_id += 1
            elif operation == "UPDATE":
                if target_id in new_exps:
                    new_exps[target_id] = content
                else:
                    new_exps[f"G{max_id}"] = content
                    max_id += 1
            elif operation == "DELETE":
                if target_id in new_exps:
                    del new_exps[target_id]
        return new_exps, revision_plan

