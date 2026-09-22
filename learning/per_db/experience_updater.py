"""DB-specific variant of ExperienceUpdater.

Uses dedicated prompt templates (DB_SPECIFIC_ROLLOUT_SUMMARY_SP/UP and
DB_SPECIFIC_GROUP_ADVANTAGE_SP/UP) that scope every distillation call to a
single database.  The prompts are told:

- Stage 1 (rollout summary): to add a "DB-Specific Observations" section that
  names exact columns, notes null/type patterns, and flags schema quirks seen
  in the trajectory.
- Stage 2 (group advantage): that the resulting experiences are *exclusively*
  used when the agent queries *this* database, so they may (and should) reference
  exact table/column names, known data quality issues, join key conventions, etc.

All other stages (3 per-group update operations, 4 batch consolidation) are
inherited unchanged from ExperienceUpdater, because those stages deal with
managing the experience pool rather than extracting content from trajectories.

The db_key is derived from the first rollout in the batch (guaranteed by
_distill_per_db to be a homogeneous db partition).
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import PracticeConfig
from ..experience_updater import ExperienceUpdater, _fmt_exec_profile, _load_prompts
from ..rollout_manager import RolloutResult
from ..utils import distill_bar as _distill_bar
from utils.llm import chat_complete_async

logger = logging.getLogger(__name__)


def _infer_db_key(rollouts: list[RolloutResult]) -> str:
    """Derive a db identifier from the first rollout's sample metadata.

    Priority: meta["db"] > stem of meta["db_path"] > "default".
    """
    if not rollouts:
        return "default"
    sample = rollouts[0].sample
    meta: dict = sample.meta or {}
    raw = ""
    if meta.get("db"):
        raw = str(meta["db"])
    elif meta.get("db_path"):
        raw = Path(str(meta["db_path"])).stem
    else:
        raw = "default"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
    return safe or "default"


class DbSpecificExperienceUpdater(ExperienceUpdater):
    """ExperienceUpdater that injects the target db name into all prompts.

    Stages 1 and 2 are overridden to use DB_SPECIFIC_* prompt templates;
    stages 3 and 4 fall through to the parent implementation unchanged.
    """

    def __init__(self, cfg: PracticeConfig) -> None:
        super().__init__(cfg)
        # Reload to pick up DB_SPECIFIC_* keys (parent already called this, but
        # we keep it explicit so this class is self-contained).
        self.prompts = _load_prompts()

    async def _single_rollout_summary(
        self,
        rollouts: list[RolloutResult],
        *,
        given_ground_truth: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        """Stage-1 override: adds db_key to the summary prompt.

        Rewrites the system prompt to use DB_SPECIFIC_ROLLOUT_SUMMARY_SP
        (falls back to SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP when the key is
        absent so existing deployments without the new prompt YAML are not
        broken).  The user prompt is analogously replaced with
        DB_SPECIFIC_ROLLOUT_SUMMARY_UP.

        The db_key is inferred from the rollout batch (all rollouts are
        expected to be from the same database, as guaranteed by _distill_per_db
        in DbSpecificTrainingFreeGRPO).
        """
        import math

        db_key = _infer_db_key(rollouts)

        sp_template = self.prompts.get(
            "DB_SPECIFIC_ROLLOUT_SUMMARY_SP",
            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP"],
        )
        up_template = self.prompts.get(
            "DB_SPECIFIC_ROLLOUT_SUMMARY_UP",
            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP"],
        )

        w_cost: float = self.cfg.evaluation.reward_w_cost
        w_token: float = self.cfg.evaluation.reward_w_token
        w_steps: float = self.cfg.evaluation.reward_w_steps

        problems_to_rollouts = self._group_rollouts_by_question(rollouts)

        all_rollouts_to_process: list[tuple[RolloutResult, float, float, float, float]] = []
        for rs in problems_to_rollouts.values():
            perfs = [float(each.reward or 0.0) for each in rs]
            if not perfs:
                continue

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
                    std = 1e-8
                return [(v - mean) / std for v in values]

            combined_advs = _z_scores(final_rewards)
            perf_advs = _z_scores(perfs)
            eff_advs = _z_scores([-c for c in cost_ranks])

            for r, fr, adv, padv, eadv in zip(rs, final_rewards, combined_advs, perf_advs, eff_advs):
                all_rollouts_to_process.append((r, fr, adv, padv, eadv))

        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sem = asyncio.Semaphore(self._distillation_concurrency)

        async def summarize_one(
            r: RolloutResult,
            final_reward: float,
            advantage: float,
            perf_adv: float,
            eff_adv: float,
        ) -> None:
            async with sem:
                try:
                    sp = (
                        sp_template
                        .replace("{{ agent_objective }}", self.agent_objective)
                        .replace("{{ learning_objective }}", self.learning_objective)
                        .replace("{{ db_key }}", db_key)
                        .replace("{{ advantage }}", f"{advantage:+.3f}")
                    )

                    traj_json = json.dumps(r.trajectory or [], ensure_ascii=False, indent=2)

                    profile_lines: list[str] = []
                    if r.exec_profile:
                        profile_lines.append("Execution Profile (one entry per MCP tool call):")
                        for i, step in enumerate(r.exec_profile, 1):
                            parts = [f"{i}. {step['tool']}"]
                            if step.get("duration_ms") is not None:
                                parts.append(f"{step['duration_ms']}ms")
                            if step.get("op_preview"):
                                parts.append(f"op: {step['op_preview']}")
                            inp_rows = step.get("input_rows")
                            out_rows = step.get("row_count")
                            if inp_rows is not None and out_rows is not None:
                                parts.append(f"in: {inp_rows} → out: {out_rows} rows")
                            elif inp_rows is not None:
                                parts.append(f"in: {inp_rows} rows")
                            elif out_rows is not None:
                                parts.append(f"out: {out_rows} rows")
                            llm_in = step.get("llm_input_tokens")
                            llm_out = step.get("llm_output_tokens")
                            if llm_in is not None or llm_out is not None:
                                parts.append(f"tokens: {llm_in or 0} in / {llm_out or 0} out")
                            if step.get("error"):
                                parts.append(f"[ERROR: {step['error']}]")
                            line = "  " + " | ".join(parts)
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
                    profile_text = (
                        "\n".join(profile_lines)
                        if profile_lines
                        else "(no execution profile available)"
                    )

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
                                "Could not read agent output CSV for sample %s; "
                                "distillation will proceed without it.",
                                r.sample.id,
                            )

                    up = (
                        up_template
                        .replace("{{ db_key }}", db_key)
                        .replace("{{ question }}", r.sample.question)
                        .replace("{{ trajectory }}", f"{profile_text}\n\nTrajectory:\n{traj_json}")
                        .replace(
                            "{{ answer }}",
                            json.dumps(r.sample.answer)
                            if (given_ground_truth and r.sample.answer is not None)
                            else "[REDACTED]",
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
                        "DB-specific rollout summary failed for sample %s (db=%s)",
                        r.sample.id,
                        db_key,
                    )

        tasks = [
            asyncio.create_task(summarize_one(r, fr, adv, padv, eadv))
            for r, fr, adv, padv, eadv in all_rollouts_to_process
        ]
        if tasks:
            bar = _distill_bar(len(tasks), f"[{db_key}] Summarise")
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
        """Stage-2 override: uses DB_SPECIFIC_GROUP_ADVANTAGE_SP/UP.

        The db_key is extracted from the first rollout in the batch so the
        distillation prompt can name the target database explicitly, anchoring
        the extracted experiences to its schema.
        """
        sp_template = self.prompts.get(
            "DB_SPECIFIC_GROUP_ADVANTAGE_SP",
            self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_SP"],
        )
        up_template = self.prompts.get(
            "DB_SPECIFIC_GROUP_ADVANTAGE_UP",
            self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_UP"],
        )

        # Derive db_key from any rollout in the batch (all are same db).
        db_key = "default"
        for rs in problem_to_summarized_rollouts.values():
            if rs:
                meta = (rs[0].get("sample") or {}).get("meta") or {}  # type: ignore[attr-defined]
                if not meta:
                    # The rollout dicts from stage-1 contain raw_question but not
                    # the full sample object.  Fall back to the class-level inference.
                    break
                if meta.get("db"):
                    db_key = str(meta["db"])
                elif meta.get("db_path"):
                    db_key = Path(str(meta["db_path"])).stem
                break

        # If we could not get it from the dict, rely on self._current_db_key
        # which is set by the run() call site below (see note in run()).
        db_key = getattr(self, "_current_db_key", db_key)

        all_rollouts: list[list[dict]] = []
        for rs in problem_to_summarized_rollouts.values():
            scores = [float(each["reward"]) for each in rs]
            if not scores:
                continue
            if len(scores) > 1:
                avg = sum(scores) / len(scores)
                var = sum((s - avg) ** 2 for s in scores) / len(scores)
                if var < 1e-8:
                    continue
            all_rollouts.append(rs)

        sem = asyncio.Semaphore(self._distillation_concurrency)
        results: list[dict] = []

        async def per_problem(rollouts_per_problem: list[dict]) -> None:
            async with sem:
                try:
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
                        sp_template
                        .replace("{{ agent_objective }}", self.agent_objective)
                        .replace("{{ learning_objective }}", self.learning_objective)
                        .replace("{{ num_experiences }}", str(num_experiences))
                        .replace("{{ db_key }}", db_key)
                    )
                    up = (
                        up_template
                        .replace("{{ db_key }}", db_key)
                        .replace("{{ question }}", rollouts_per_problem[0]["raw_question"])
                        .replace(
                            "{{ answer }}",
                            json.dumps(rollouts_per_problem[0]["correct_answer"])
                            if given_ground_truth
                            else "[REDACTED]",
                        )
                        .replace("{{ trajectories }}", formatted)
                    )
                    response = await chat_complete_async(
                        self.model, system=sp, user=up, temperature=0.2
                    )
                    pattern = re.compile(
                        r"<Experiences>\s*(.*?)\s*</Experiences>",
                        re.DOTALL | re.IGNORECASE,
                    )
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
                        "DB-specific group advantage failed for question: %.80s... (db=%s)",
                        rollouts_per_problem[0].get("raw_question", ""),
                        db_key,
                    )

        tasks = [asyncio.create_task(per_problem(rs)) for rs in all_rollouts]
        if tasks:
            bar = _distill_bar(len(tasks), f"[{db_key}] Grp adv   ")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results
