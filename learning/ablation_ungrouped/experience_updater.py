"""Ablation variant of ExperienceUpdater: independent rollout distillation.

Instead of computing group-relative z-score advantages (GRPO), each rollout is
treated in isolation.  Stages 3 (per-group experience update) and 4 (batch
consolidation) are inherited unchanged from ExperienceUpdater.

Key differences from the GRPO variant:
- _single_rollout_summary: reports absolute performance + cost scores only;
  no group z-score is computed (advantage = raw final_reward).
- _group_advantage: every rollout is distilled independently — one LLM call per
  rollout rather than one call that compares all rollouts for the same question.
  The zero-variance skip (which would suppress all-identical groups) does not
  apply here because there is no group to compare within.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from collections import defaultdict
from typing import Any

from ..config import PracticeConfig
from ..experience_updater import ExperienceUpdater, _fmt_exec_profile, _load_prompts
from ..rollout_manager import RolloutResult
from ..utils import TaskRecorder, distill_bar as _distill_bar
from utils.llm import chat_complete_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt keys added to prompts/experience.yaml for the independent variant.
# If they are absent we fall back to adapted versions of the GRPO prompts.
# ---------------------------------------------------------------------------
_INDEP_SUMMARY_SP_KEY = "INDEP_ROLLOUT_SUMMARY_TEMPLATE_SP"
_INDEP_SUMMARY_UP_KEY = "INDEP_ROLLOUT_SUMMARY_TEMPLATE_UP"
_INDEP_SINGLE_SP_KEY  = "INDEP_SINGLE_ROLLOUT_DISTILL_SP"
_INDEP_SINGLE_UP_KEY  = "INDEP_SINGLE_ROLLOUT_DISTILL_UP"


class IndependentExperienceUpdater(ExperienceUpdater):
    """GRPO-free experience updater: distills each rollout independently.

    Rollouts are still batched and run concurrently, but the advantage signal
    fed to the distillation LLM is the *absolute* performance-minus-cost score
    rather than a group-normalised z-score.  This is the ablation baseline that
    removes the group-relative comparison from the full GRPO pipeline.
    """

    def __init__(self, cfg: PracticeConfig) -> None:
        super().__init__(cfg)
        # Re-load prompts so the independent-specific keys are available.
        self.prompts = _load_prompts()

    # ------------------------------------------------------------------
    # Stage 1 override: compute per-rollout absolute scores (no z-score)
    # ------------------------------------------------------------------

    async def _single_rollout_summary(
        self,
        rollouts: list[RolloutResult],
        *,
        given_ground_truth: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        """Summarise each rollout individually using absolute scores only.

        Unlike the GRPO variant, no within-group normalisation is performed and
        no z-scores are computed.  The reported 'advantage' is simply the raw
        combined final_reward (performance minus absolute cost penalty).

        Cost is normalised using a fixed reference scale (10 000 LLM-op tokens
        and 20 steps) so the value is always in a stable [0, 1] range even when
        only a single rollout exists.  This avoids the degenerate single-sample
        z-score issue while keeping costs comparable across runs.
        """
        w_cost: float = self.cfg.evaluation.reward_w_cost
        w_token: float = self.cfg.evaluation.reward_w_token
        w_steps: float = self.cfg.evaluation.reward_w_steps

        # Reference scale for absolute cost normalisation.
        _TOKEN_REF = 10_000   # tokens above which cost_rank_token saturates at 1
        _STEPS_REF = 20       # steps above which cost_rank_steps saturates at 1

        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sem = asyncio.Semaphore(self._distillation_concurrency)

        async def summarize_one(r: RolloutResult) -> None:
            async with sem:
                try:
                    perf = float(r.reward or 0.0)

                    # Absolute cost normalisation (clipped to [0, 1]).
                    token_rank = min(r.llmop_tokens / _TOKEN_REF, 1.0)
                    step_rank  = min(r.num_steps   / _STEPS_REF,  1.0)
                    cost_rank  = w_token * token_rank + w_steps * step_rank
                    final_reward = perf - w_cost * cost_rank

                    # --- Build trajectory + exec-profile text ---
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
                            llm_in  = step.get("llm_input_tokens")
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
                                data   = csv_rows[1:]
                                lines  = [", ".join(header)]
                                for row in data:
                                    lines.append(", ".join(row))
                                agent_output_text = "\n".join(lines)
                        except Exception:
                            logger.warning(
                                "Could not read agent output CSV for sample %s at %s.",
                                r.sample.id, r.output_csv_path,
                            )

                    # Choose prompt template: use independent-specific keys when
                    # available, otherwise fall back to the GRPO summary template
                    # with the advantage label overridden to "Score (absolute)".
                    if _INDEP_SUMMARY_SP_KEY in self.prompts:
                        sp = (
                            self.prompts[_INDEP_SUMMARY_SP_KEY]
                            .replace("{{ agent_objective }}", self.agent_objective)
                            .replace("{{ learning_objective }}", self.learning_objective)
                        )
                        up = (
                            self.prompts[_INDEP_SUMMARY_UP_KEY]
                            .replace("{{ question }}", r.sample.question)
                            .replace(
                                "{{ trajectory }}",
                                f"{profile_text}\n\nTrajectory:\n{json.dumps(r.trajectory or [], ensure_ascii=False, indent=2)}",
                            )
                            .replace(
                                "{{ answer }}",
                                json.dumps(r.sample.answer)
                                if (given_ground_truth and r.sample.answer is not None)
                                else "[REDACTED]",
                            )
                            .replace("{{ critique }}", r.reasoning or "[No critique provided]")
                            .replace("{{ agent_output }}", agent_output_text)
                            .replace("{{ perf_reward }}", f"{perf:.4f}")
                            .replace("{{ final_reward }}", f"{final_reward:.4f}")
                            .replace("{{ llmop_tokens }}", str(r.llmop_tokens))
                            .replace("{{ num_steps }}", str(r.num_steps))
                        )
                    else:
                        # Fallback: reuse GRPO summary template, replacing the
                        # group-relative advantage label with an absolute score note.
                        sp = (
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP"]
                            .replace("{{ agent_objective }}", self.agent_objective)
                            .replace("{{ learning_objective }}", self.learning_objective)
                        )
                        up = (
                            self.prompts["SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP"]
                            .replace("{{ question }}", r.sample.question)
                            .replace(
                                "{{ trajectory }}",
                                f"{profile_text}\n\nTrajectory:\n{json.dumps(r.trajectory or [], ensure_ascii=False, indent=2)}",
                            )
                            .replace(
                                "{{ answer }}",
                                json.dumps(r.sample.answer)
                                if (given_ground_truth and r.sample.answer is not None)
                                else "[REDACTED]",
                            )
                            .replace("{{ critique }}", r.reasoning or "[No critique provided]")
                            .replace("{{ agent_output }}", agent_output_text)
                            .replace("{{ perf_reward }}", f"{perf:.4f}")
                            # Replace the group-relative advantage with an absolute score.
                            .replace(
                                "{{ advantage }}",
                                f"{final_reward:+.4f} (absolute score; no group comparison)",
                            )
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
                            "perf_reward": perf,
                            "llmop_tokens": r.llmop_tokens,
                            "num_steps": r.num_steps,
                            "reward": final_reward,
                            # advantage == final_reward (absolute); no z-score.
                            "advantage": final_reward,
                            "perf_adv": perf,
                            "eff_adv": -(w_token * token_rank + w_steps * step_rank),
                            "trajectory_summary": response,
                        }
                    )
                except Exception:
                    logger.exception(
                        "Independent rollout summary failed for sample %s", r.sample.id
                    )

        tasks = [asyncio.create_task(summarize_one(r)) for r in rollouts]
        if tasks:
            bar = _distill_bar(len(tasks), "Summarise rollouts")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results

    # ------------------------------------------------------------------
    # Stage 2 override: distill each rollout independently (no grouping)
    # ------------------------------------------------------------------

    async def _group_advantage(
        self,
        problem_to_summarized_rollouts: dict[str, list[dict]],
        *,
        given_ground_truth: bool,
        num_experiences: int,
    ) -> list[dict]:
        """Distil experiences from each rollout independently (no group comparison).

        Each rollout summary is passed to the LLM as a standalone trajectory —
        the LLM sees only one rollout at a time and extracts what worked or
        what failed without any cross-rollout comparative signal.  The zero-
        variance group skip that the GRPO variant uses does not apply here.
        """
        # Flatten all rollouts from all questions into a single list.
        all_single_rollouts: list[dict] = []
        for rs in problem_to_summarized_rollouts.values():
            all_single_rollouts.extend(rs)

        sem = asyncio.Semaphore(self._distillation_concurrency)
        results: list[dict] = []

        async def distill_one(entry: dict) -> None:
            async with sem:
                try:
                    score_label = (
                        f"Score {entry['reward']:+.4f} (absolute) "
                        f"| Correctness {entry['perf_reward']:.4f} "
                        f"| Steps {entry['num_steps']} "
                        f"| LLMOp tokens {entry['llmop_tokens']}"
                    )

                    if _INDEP_SINGLE_SP_KEY in self.prompts:
                        sp = (
                            self.prompts[_INDEP_SINGLE_SP_KEY]
                            .replace("{{ agent_objective }}", self.agent_objective)
                            .replace("{{ learning_objective }}", self.learning_objective)
                            .replace("{{ num_experiences }}", str(num_experiences))
                        )
                        up = (
                            self.prompts[_INDEP_SINGLE_UP_KEY]
                            .replace("{{ question }}", entry["raw_question"])
                            .replace(
                                "{{ answer }}",
                                json.dumps(entry["correct_answer"])
                                if given_ground_truth
                                else "[REDACTED]",
                            )
                            .replace("{{ trajectory }}", entry["trajectory_summary"])
                            .replace("{{ score_label }}", score_label)
                        )
                    else:
                        # Fallback: adapt the GRPO group-advantage prompt to a
                        # single-rollout context by replacing group-comparison
                        # language with single-rollout language.
                        sp = (
                            self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_SP"]
                            .replace("{{ agent_objective }}", self.agent_objective)
                            .replace("{{ learning_objective }}", self.learning_objective)
                            .replace("{{ num_experiences }}", str(num_experiences))
                        )
                        trajectory_block = (
                            f"Attempt 1 ({score_label}):\n"
                            f"{entry['trajectory_summary']}"
                        )
                        up = (
                            self.prompts["SINGLE_QUERY_GROUP_ADVANTAGE_UP"]
                            .replace("{{ question }}", entry["raw_question"])
                            .replace(
                                "{{ answer }}",
                                json.dumps(entry["correct_answer"])
                                if given_ground_truth
                                else "[REDACTED]",
                            )
                            .replace("{{ trajectories }}", trajectory_block)
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
                    # Wrap in the same dict shape expected by _group_update.
                    results.append(
                        {
                            "rollouts": [entry],
                            "critique": response,
                            "experiences": experiences,
                        }
                    )
                except Exception:
                    logger.exception(
                        "Independent rollout distillation failed for sample %s",
                        entry.get("sample_id", "?"),
                    )

        tasks = [asyncio.create_task(distill_one(e)) for e in all_single_rollouts]
        if tasks:
            bar = _distill_bar(len(tasks), "Indep distill    ")
            try:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
            finally:
                bar.close()
        return results
