"""Rollout manager that drives the DB agent for training-free GRPO."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .config import PracticeConfig
from .data_manager import JsonlDataManager, PracticeSample
from .utils import TaskRecorder, rollout_bar as _rollout_bar, status_bar as _status_bar

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Single rollout result for a (query, attempt) pair."""

    sample: PracticeSample
    final_answer: str
    trajectory: list[dict[str, Any]]
    reward: float                          # performance score ∈ [0, 1]
    reasoning: str | None
    llmop_tokens: int = 0                  # LLM-op tokens; used for within-group cost normalisation
    num_steps: int = 0                     # total MCP tool calls (excl. session mgmt); used for cost normalisation
    token_usage: dict[str, Any] | None = None
    output_csv_path: str | None = None
    # Compact per-step execution profile from MCP server logs.
    # Each entry: {tool, duration_ms, input_rows, row_count, llm_input_tokens, llm_output_tokens, error}
    # input_rows is populated for LLM ops only; row_count is the output cardinality for all ops.
    exec_profile: list[dict[str, Any]] | None = None
    # Plan-enumeration metadata threaded from PracticeSample for distillation labeling.
    plan_hint: str | None = None
    plan_axes: dict[str, Any] | None = None
    # True when the rollout hit task_timeout; the trajectory is partial.
    timed_out: bool = False


class RolloutManager:
    """Minimal analogue of `utu.practice.RolloutManager` for the DB agent.

    Responsibilities:
    - Create per-epoch duplicated samples via JsonlDataManager
    - For each batch:
        - Run the DB agent multiple times (once per duplicated sample)
        - Score each attempt using a user-provided verifier
    """

    def __init__(
        self,
        config: PracticeConfig,
        *,
        run_agent_once: Callable[..., Awaitable[dict[str, Any]]],
        verify_func: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.config = config
        self.data_manager = JsonlDataManager(config)
        self.run_agent_once = run_agent_once
        self.verify_func = verify_func
        self._enumerator: Any | None = None   # lazy-initialised in load_epoch_data_async

    def load_epoch_data(self, epoch: int) -> list[PracticeSample]:
        """Load epoch samples without plan-hint assignment.

        Called internally by :meth:`load_epoch_data_async`.  External callers
        should use ``load_epoch_data_async`` to get plan-enumeration support.
        """
        return self.data_manager.load_epoch_data(
            epoch,
            truncate=self.config.practice.rollout_data_truncate,
            shuffle=self.config.practice.shuffle_data,
        )

    async def load_epoch_data_async(self, epoch: int) -> list[PracticeSample]:
        """Like :meth:`load_epoch_data` but also assigns plan hints when plan_enumeration is enabled.

        For each unique base question, calls PlanEnumerator.select once to obtain k
        axis-value assignments and assembles them into strategy hint strings.  Each
        duplicated PracticeSample receives exactly one hint in round-robin order.

        Falls back to :meth:`load_epoch_data` (no hints) on any error so training
        always makes progress.
        """
        samples = self.load_epoch_data(epoch)

        if not getattr(self.config.practice, "plan_enumeration", False):
            return samples

        from .plan_enumerator import PlanEnumerator, load_plan_library

        if self._enumerator is None:
            lib_path = getattr(self.config.practice, "plan_library_path", None)
            lib = load_plan_library(lib_path)
            self._enumerator = PlanEnumerator(lib)

        enumerator: PlanEnumerator = self._enumerator
        k = self.config.practice.grpo_n

        # Group duplicated samples by base question id (strip "::attempt_N" suffix).
        from collections import defaultdict as _dd
        groups: dict[str, list[PracticeSample]] = _dd(list)
        for s in samples:
            base_id = s.id.split("::")[0]
            groups[base_id].append(s)

        # Enumerate plans for each unique base question concurrently.
        sem = asyncio.Semaphore(4)   # limit concurrent DB reads + LLM calls

        async def _assign_group(base_id: str, group: list[PracticeSample]) -> None:
            async with sem:
                db_path = self._resolve_db_path(group[0])
                question = group[0].question
                try:
                    hints, axes_list = await enumerator.select(question, db_path, k)
                except Exception:
                    logger.warning(
                        "PlanEnumerator failed for %s; falling back to RandomPlanEnumerator.",
                        base_id,
                    )
                    from .plan_enumerator import RandomPlanEnumerator
                    rand_enum = RandomPlanEnumerator(lib)
                    hints, axes_list = await rand_enum.select(question, db_path, k)
                for i, s in enumerate(group):
                    idx = i % len(hints)
                    s.plan_hint = hints[idx]
                    s.plan_axes = dict(axes_list[idx])

        tasks = [
            asyncio.create_task(_assign_group(base_id, group))
            for base_id, group in groups.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)

        return samples

    def _resolve_db_path(self, s: PracticeSample) -> str:
        """Return the DuckDB path to use for a given sample.

        Priority:
          1. ``meta["db_path"]``  — explicit absolute/relative path in the data
          2. ``meta["db"]``       — bare db name resolved against ``config.db_files_dir``
        """
        meta: dict = s.meta or {}

        if meta.get("db_path"):
            return str(meta["db_path"])

        db_name: str | None = meta.get("db")
        if db_name:
            from pathlib import Path as _Path
            db_files_dir: str | None = getattr(self.config, "db_files_dir", None)
            if db_files_dir:
                candidate = _Path(db_files_dir) / f"{db_name}.duckdb"
                if candidate.exists():
                    return str(candidate)
            raise ValueError(
                f"Cannot resolve db '{db_name}' for sample '{s.id}': "
                f"DB_FILES_DIR is not set or '{db_name}.duckdb' not found in it. "
                "Set DB_FILES_DIR in .env to the directory containing your .duckdb files."
            )

        raise ValueError(
            f"Sample '{s.id}' has no 'db' or 'db_path' field and no global fallback is set. "
            "Add a 'db' field to each sample or set 'db_path' explicitly in the sample metadata."
        )

    async def run_batch(
        self,
        *,
        epoch_samples: list[PracticeSample],
        batch_idx: int,
        queries_per_update: int | None = None,
        recorder: TaskRecorder | None = None,
        desc: str = "",
    ) -> list[RolloutResult]:
        effective_qpu = queries_per_update if queries_per_update is not None else self.config.practice.queries_per_update
        batch = self.data_manager.get_batch(
            epoch_samples,
            batch_idx=batch_idx,
            grpo_n=self.config.practice.grpo_n,
            queries_per_update=effective_qpu,
        )
        if not batch:
            return []

        sem = asyncio.Semaphore(self.config.practice.rollout_concurrency)
        task_timeout = self.config.practice.task_timeout
        max_retries = self.config.practice.max_retries

        # Per-rollout status bars: each active rollout gets one line below the
        # main progress bar.  We allocate positions from a thread-safe free list.
        import threading
        concurrency = self.config.practice.rollout_concurrency
        _slot_lock = threading.Lock()
        _free_slots: list[int] = list(range(concurrency))
        # position offset: 1 (main bar) + 1 (batch bar) + 1 (rollout bar) = base 3
        _BAR_BASE = 3

        async def _run_one(s: PracticeSample, experiences_text: str | None) -> RolloutResult | None:
            async with sem:
                # Grab a display slot for this rollout's status bar.
                with _slot_lock:
                    slot = _free_slots.pop(0) if _free_slots else 0
                short_id = s.id.split("::")[0]
                status_bar = _status_bar(short_id, position=_BAR_BASE + slot)
                status_bar.set_postfix_str("starting…")

                def _on_action(tool_name: str, action_count: int) -> None:
                    status_bar.set_postfix_str(f"#{action_count} {tool_name}")

                sample_db_path = self._resolve_db_path(s)
                try:
                    for attempt in range(max_retries):
                        try:
                            agent_out = await self.run_agent_once(
                                sample_db_path, s.question, experiences_text,
                                on_action=_on_action,
                                plan_hint=s.plan_hint,
                                task_timeout=task_timeout,
                            )
                            timed_out: bool = bool(agent_out.get("timed_out"))
                            if timed_out:
                                logger.info(
                                    "Rollout timed out after %ds for sample %s; "
                                    "using partial trajectory (no retry).",
                                    task_timeout, s.id,
                                )
                            final_answer = str(agent_out.get("final_answer", "")) or ""
                            trajectory = list(agent_out.get("trajectory") or [])
                            token_usage: dict[str, Any] = dict(agent_out.get("token_usage") or {})
                            output_csv_path: str | None = agent_out.get("output_csv_path")
                            exec_profile: list[dict[str, Any]] | None = agent_out.get("exec_profile") or None
                            verify_input = {
                                "id": s.id,
                                "question": s.question,
                                "answer": s.answer,
                                "final_answer": final_answer,
                                "output_csv_path": output_csv_path,
                                "trajectory": trajectory,
                                "token_usage": token_usage,
                                "meta": s.meta,
                            }
                            vr = self.verify_func(verify_input)
                            reward = float(vr.get("reward", 0.0))
                            llmop_tokens = int(vr.get("llmop_tokens", 0))
                            reasoning = vr.get("reasoning")
                            # exec_profile already excludes open/close_session entries.
                            num_steps = len(exec_profile) if exec_profile else 0
                            s.stage = "judged"
                            s.trajectories = trajectory
                            s.reward = reward
                            s.reasoning = reasoning
                            return RolloutResult(
                                sample=s,
                                final_answer=final_answer,
                                trajectory=trajectory,
                                reward=reward,
                                reasoning=reasoning,
                                llmop_tokens=llmop_tokens,
                                num_steps=num_steps,
                                token_usage=token_usage,
                                output_csv_path=output_csv_path,
                                exec_profile=exec_profile,
                                plan_hint=s.plan_hint,
                                plan_axes=s.plan_axes,
                                timed_out=timed_out,
                            )
                        except Exception:
                            logger.exception(
                                "Rollout failed for sample %s (attempt %d/%d)",
                                s.id, attempt + 1, max_retries,
                            )
                            status_bar.set_postfix_str(f"error (attempt {attempt + 1})")
                    logger.error("All %d attempts failed for sample %s; dropping from batch.", max_retries, s.id)
                    return None
                finally:
                    status_bar.close()
                    with _slot_lock:
                        _free_slots.append(slot)

        # Format current experiences (if any) as a numbered list, similar to the original
        # Training-Free GRPO implementation.
        experiences_text: str | None = None
        if recorder is not None and recorder.experiences:
            experiences_text = "\n".join(f"[{k}]. {v}" for k, v in recorder.experiences.items())

        tasks = [_run_one(s, experiences_text) for s in batch]
        results: list[RolloutResult] = []
        with _rollout_bar(len(tasks)) as pbar:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                if r is not None:
                    results.append(r)
                    pbar.set_postfix_str(f"reward={r.reward:.3f}  done={len(results)}")
                pbar.update(1)

        if recorder is not None:
            rewards = [r.reward for r in results]
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                total_llmop = sum(r.llmop_tokens for r in results)
                recorder.stat_update(
                    {
                        f"batch_{batch_idx}": {
                            "num_samples": len(results),
                            "mean_reward": mean_reward,
                            "total_llmop_tokens": total_llmop,
                        }
                    }
                )
                logger.info(
                    "Batch %d complete: %d/%d rollouts succeeded  "
                    "mean_reward=%.3f  llmop_tokens=%d",
                    batch_idx, len(results), len(tasks), mean_reward, total_llmop,
                )
        return results


