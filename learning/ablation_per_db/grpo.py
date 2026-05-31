"""DB-specific variant of training-free GRPO.

Each database gets its own experience pool.  At rollout time, an agent query
only receives experiences that were distilled from the same database.  The
underlying GRPO distillation (group-relative z-score advantages, 4-stage
pipeline) is unchanged; it is simply run once per database partition rather
than once over the full mixed batch.

New files (nothing existing is modified):
  learning/ablation_per_db/grpo.py  -- this file
  learning/ablation_per_db/cli.py   -- CLI entrypoint

Design overview
---------------
1. DB key extraction
   Every PracticeSample carries meta["db"] (bare name, e.g. "california_schools")
   or meta["db_path"] (full path).  _db_key() converts either to a safe string.

2. Per-db experience store
   self._db_experiences: dict[str, dict[str, str]]
   Maps db_key -> {exp_id -> text}.  The parent recorder.experiences is not used
   for injection; it is kept None so RolloutManager passes experiences_text=None
   to run_agent_once.

3. Experience injection at rollout time
   TrainingFreeDbSpecific passes a *wrapper* for run_agent_once to RolloutManager.
   The wrapper receives db_path, ignores the (None) experiences_text argument the
   manager built, and looks up the correct db-specific pool instead.

4. Distillation per db
   After each batch, rollouts are partitioned by db_key.  For each partition a
   temporary TaskRecorder is created with that db's current pool, and
   ExperienceUpdater.run() is called normally.  The standard GRPO group-relative
   advantages are computed within the db partition.  Results are stored back
   into self._db_experiences[db_key].

5. Caching
   Rollouts: stored under <exp_dir>/rollout_cache/ (same as base class).
   DB experiences: stored under <exp_dir>/experiences/ and
   <run_log_dir>/experiences/ (same dual-write pattern as base class).

6. Experienced-prompt output
   One prompt file per db: <run_log_dir>/experienced_prompt_<db_key>.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import PracticeConfig
from ..data_manager import PracticeSample
from .experience_updater import DbSpecificExperienceUpdater
from ..rollout_manager import RolloutManager, RolloutResult
from ..enumgrpo import TrainingFreeGRPO, TrainingCheckpoint
from ..utils import (
    TaskRecorder,
    epoch_bar as _epoch_bar,
    batch_bar as _batch_bar,
    tqdm_write as _tqdm_write,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB key helpers
# ---------------------------------------------------------------------------

def _db_key(s: PracticeSample) -> str:
    """Return a clean db identifier for a sample.

    Priority: meta["db"] > stem of meta["db_path"] > "default".
    Non-alphanumeric characters are replaced with underscores so the key is
    safe to use in file names and dict lookups.
    """
    meta: dict = s.meta or {}
    raw: str = ""
    if meta.get("db"):
        raw = str(meta["db"])
    elif meta.get("db_path"):
        raw = Path(str(meta["db_path"])).stem
    else:
        raw = "default"
    # Sanitise: keep letters, digits, hyphens, underscores.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
    return safe or "default"


def _db_key_from_path(db_path: str) -> str:
    """Derive a db key directly from a filesystem path (used in the wrapper)."""
    return Path(db_path).stem


# ---------------------------------------------------------------------------
# DB-specific experience cache helpers
# ---------------------------------------------------------------------------

def _db_experiences_filename(step: int) -> str:
    return f"step_{step:04d}_db.json"


def _save_db_experiences(
    run_dir: Path,
    exp_dir: Path,
    step: int,
    db_experiences: dict[str, dict[str, str]],
    *,
    epoch: int | None = None,
    batch: int | None = None,
) -> None:
    """Write per-db experiences to both the run-level and experiment-level dirs."""
    payload = {
        "step": step,
        "epoch": epoch,
        "batch": batch,
        "db_experiences": db_experiences,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    fname = _db_experiences_filename(step)

    for target_dir in (run_dir / "experiences", exp_dir / "experiences"):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / fname).write_text(text, encoding="utf-8")
        (target_dir / "latest_db.json").write_text(text, encoding="utf-8")


def _load_db_experiences(
    exp_dir: Path,
    step: int,
) -> dict[str, dict[str, str]] | None:
    """Load per-db experiences from the experiment-level dir (canonical for resume)."""
    path = exp_dir / "experiences" / _db_experiences_filename(step)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Corrupt db_experiences file at %s; will re-run step.", path)
        return None
    db_exps = raw.get("db_experiences")
    if not isinstance(db_exps, dict):
        logger.warning("Invalid db_experiences file at %s; will re-run step.", path)
        return None
    return db_exps


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class DbSpecificTrainingFreeGRPO(TrainingFreeGRPO):
    """Training-free GRPO with per-database experience pools.

    All rollout mechanics (plan enumeration, concurrency, caching, checkpointing)
    and the distillation algorithm (group-relative GRPO) are inherited unchanged.
    The only difference is that experiences are segregated by database: at
    inference/rollout time an agent receives only experiences from its own db,
    and distillation runs independently per db partition.
    """

    def __init__(self, config: PracticeConfig, *, run_log_dir: Path | None = None) -> None:
        super().__init__(config, run_log_dir=run_log_dir)
        # Per-db experience pools: db_key -> {exp_id -> text}.
        self._db_experiences: dict[str, dict[str, str]] = {}
        # Replace the generic updater with the DB-specific one so that all
        # distillation prompts are scoped to the target database.
        self._experience_updater = DbSpecificExperienceUpdater(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_experiences_text(self, db_key: str) -> str | None:
        """Return formatted experience text for a given db_key, or None."""
        pool = self._db_experiences.get(db_key) or {}
        if not pool:
            return None
        return "\n".join(f"[{k}]. {v}" for k, v in pool.items())

    def _make_db_aware_run_agent_once(self):
        """Return a wrapper around _run_agent_once that injects db-specific experiences.

        RolloutManager builds experiences_text from recorder.experiences (which
        we keep empty/None).  This wrapper discards that argument and substitutes
        the correct per-db pool based on db_path.
        """
        async def _wrapper(
            db_path: str,
            question: str,
            experiences_text: str | None,   # ignored; replaced below
            on_action=None,
            plan_hint: str | None = None,
            task_timeout: float | None = None,
        ) -> dict:
            key = _db_key_from_path(db_path)
            db_exp_text = self._get_experiences_text(key)
            return await self._run_agent_once(
                db_path, question, db_exp_text,
                on_action=on_action,
                plan_hint=plan_hint,
                task_timeout=task_timeout,
            )
        return _wrapper

    # ------------------------------------------------------------------
    # Per-db distillation
    # ------------------------------------------------------------------

    async def _distill_per_db(
        self,
        rollouts: list[RolloutResult],
        *,
        step: int,
        epoch: int,
        batch: int,
    ) -> None:
        """Partition rollouts by db and run the full GRPO distillation per partition.

        Each partition gets its own temporary TaskRecorder seeded with that db's
        current experience pool.  After distillation the updated pool is stored
        back into self._db_experiences[db_key].
        """
        # Group rollouts by db_key.
        partitions: dict[str, list[RolloutResult]] = defaultdict(list)
        for r in rollouts:
            partitions[_db_key(r.sample)].append(r)

        for db_key, db_rollouts in partitions.items():
            _tqdm_write(
                f"    [step {step}] DB '{db_key}': distilling from "
                f"{len(db_rollouts)} rollouts…"
            )
            # Build a throw-away recorder seeded with this db's pool.
            mini_recorder = TaskRecorder(
                experiment_name=f"{self.config.exp_id}::{db_key}"
            )
            current_pool = self._db_experiences.get(db_key) or {}
            if current_pool:
                mini_recorder.experiences_update(current_pool)

            # Propagate the db_key to the updater so _group_advantage can
            # reference it even when the rollout dicts don't carry full sample
            # metadata.
            self._experience_updater._current_db_key = db_key  # type: ignore[attr-defined]

            try:
                new_pool = await self._experience_updater.run(
                    rollouts=db_rollouts,
                    recorder=mini_recorder,
                    log_dir=self._run_log_dir / f"db_{db_key}" if self._run_log_dir else None,
                    step=step,
                    epoch=epoch,
                    batch=batch,
                )
                self._db_experiences[db_key] = new_pool
                _tqdm_write(
                    f"    [step {step}] DB '{db_key}': {len(new_pool)} experiences."
                )
            except Exception:
                logger.exception(
                    "Distillation failed for db '%s' at step %d; "
                    "keeping previous pool (%d experiences).",
                    db_key, step, len(current_pool),
                )

    # ------------------------------------------------------------------
    # Checkpoint helpers for db-specific pools
    # ------------------------------------------------------------------

    def _checkpoint_db_experiences(self, *, step: int, epoch: int, batch: int) -> None:
        """Write the current per-db experience pools to both the run and experiment dirs."""
        if self._run_log_dir is None or not self._db_experiences:
            return
        if self._run_log_dir is not None:
            _save_db_experiences(
                self._run_log_dir,
                self._run_log_dir.parent,
                step,
                self._db_experiences,
                epoch=epoch,
                batch=batch,
            )
        logger.debug("Wrote db experience checkpoint for step %d", step)

    # ------------------------------------------------------------------
    # Experienced prompt output (one file per db)
    # ------------------------------------------------------------------

    def _write_db_experienced_prompts(
        self,
        db_experiences: dict[str, dict[str, str]],
    ) -> dict[str, Path]:
        """Write one system-prompt file per database to the run log dir.

        Returns a dict mapping db_key -> output path.
        """
        from agent.codebase.config import PROJECT_ROOT

        repo_root = Path(PROJECT_ROOT).resolve().parent
        prompts_dir = Path(PROJECT_ROOT).resolve() / "codebase" / "prompts"

        if self.config.system_prompt_path:
            base_prompt_path = Path(self.config.system_prompt_path)
            if not base_prompt_path.is_absolute():
                base_prompt_path = repo_root / base_prompt_path
        else:
            base_prompt_path = prompts_dir / "db_system_prompt.md"

        base_text = (
            base_prompt_path.read_text(encoding="utf-8")
            if base_prompt_path.exists()
            else ""
        )

        output_paths: dict[str, Path] = {}
        for db_key, pool in db_experiences.items():
            if not pool:
                continue
            exp_lines = [
                f"\n\n## Learned Experiences for Database: {db_key} (Training-Free GRPO)",
                "",
                "When solving problems on this database, you MUST first carefully read "
                "and understand the helpful instructions and experiences:",
            ]
            for k, v in sorted(pool.items()):
                exp_lines.append(f"[{k}]. {v}")

            content = base_text.rstrip() + "\n" + "\n".join(exp_lines) + "\n"

            if self._run_log_dir is not None:
                out_path = self._run_log_dir / f"experienced_prompt_{db_key}.md"
                out_path.write_text(content, encoding="utf-8")
                logger.info("Wrote db-specific experienced prompt to %s", out_path)
                output_paths[db_key] = out_path

        return output_paths

    # ------------------------------------------------------------------
    # Main training loop override
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, dict[str, str]]:  # type: ignore[override]
        """Run training-free GRPO with per-db experience segregation.

        Returns the final db_experiences dict (db_key -> experience pool).
        """
        # Build RolloutManager with the db-aware agent wrapper so that each
        # rollout receives only experiences from its own database.  Pass
        # recorder=self.recorder to run_batch for stats tracking, but keep
        # recorder.experiences = None so the manager's experience injection
        # is a no-op (the wrapper handles injection instead).
        self._rollout_manager = RolloutManager(
            self.config,
            run_agent_once=self._make_db_aware_run_agent_once(),
            verify_func=self._verify_func,
        )
        rollout_manager = self._rollout_manager

        epoch_bar = _epoch_bar(self.config.practice.epochs)

        # Pre-seed from run dir when resuming.
        restart_step = self.config.practice.restart_step
        if restart_step:
            exp_dir = self._run_log_dir.parent if self._run_log_dir else None
            if exp_dir is not None:
                for seed_step in range(restart_step - 1, -1, -1):
                    seed_db_exps = _load_db_experiences(exp_dir, seed_step)
                    if seed_db_exps is not None:
                        for dk, pool in seed_db_exps.items():
                            self._db_experiences[dk] = pool
                        _tqdm_write(
                            f"  [resume] Pre-seeded db experience pools from step {seed_step} "
                            f"({len(seed_db_exps)} dbs, "
                            f"{sum(len(p) for p in seed_db_exps.values())} total experiences)."
                        )
                        break

        new_batches_this_session = 0
        for epoch in epoch_bar:
            epoch_data = await rollout_manager.load_epoch_data_async(epoch)
            total = len(epoch_data)
            queries_per_update = min(
                self.config.practice.queries_per_update,
                total // self.config.practice.grpo_n,
            )
            if queries_per_update < 1:
                raise ValueError(
                    f"Epoch {epoch}: dataset too small ({total} expanded samples) for "
                    f"grpo_n={self.config.practice.grpo_n}."
                )
            group_size = queries_per_update * self.config.practice.grpo_n
            num_batches = math.ceil(total / group_size)

            batch_bar = _batch_bar(num_batches, epoch)
            for batch_idx in batch_bar:
                step = epoch * num_batches + batch_idx
                batch_bar.set_postfix_str(f"step={step}")

                # ----------------------------------------------------------
                # 1. Rollout phase (cached or fresh)
                # ----------------------------------------------------------
                cached_rollout_dicts = self._load_rollouts(step)
                if cached_rollout_dicts is not None and self._should_use_cache(step):
                    rollouts = self._reconstruct_rollouts(cached_rollout_dicts)
                    _tqdm_write(
                        f"  [step {step}] Loaded {len(rollouts)} rollouts from run dir."
                    )
                else:
                    rollouts = await rollout_manager.run_batch(
                        epoch_samples=epoch_data,
                        batch_idx=batch_idx,
                        queries_per_update=queries_per_update,
                        recorder=self.recorder,
                        desc=f"  E{epoch} B{batch_idx} rollouts",
                    )
                    if rollouts and self._should_use_cache(step):
                        self._save_rollouts(step, rollouts)

                if not rollouts:
                    continue

                rewards = [r.reward for r in rollouts]
                mean_r = sum(rewards) / len(rewards) if rewards else 0.0
                batch_bar.set_postfix(step=step, mean_reward=f"{mean_r:.3f}")

                # ----------------------------------------------------------
                # 2. Distillation phase (cached or fresh, per db)
                # ----------------------------------------------------------
                exp_dir = self._run_log_dir.parent if self._run_log_dir else None
                cached_db_exps = _load_db_experiences(exp_dir, step) if exp_dir else None
                if cached_db_exps is not None and self._should_use_cache(step):
                    for dk, pool in cached_db_exps.items():
                        self._db_experiences[dk] = pool
                    _tqdm_write(
                        f"  [step {step}] Loaded db experiences from run dir "
                        f"({len(cached_db_exps)} dbs)."
                    )
                else:
                    _tqdm_write(
                        f"  [step {step}] Distilling per-db experiences from "
                        f"{len(rollouts)} rollouts (mean_reward={mean_r:.3f})…"
                    )
                    await self._distill_per_db(
                        rollouts, step=step, epoch=epoch, batch=batch_idx
                    )
                    if self._run_log_dir is not None:
                        _save_db_experiences(
                            self._run_log_dir,
                            self._run_log_dir.parent,
                            step,
                            self._db_experiences,
                            epoch=epoch,
                            batch=batch_idx,
                        )
                    new_batches_this_session += 1

                # ----------------------------------------------------------
                # 3. Checkpoint artefacts
                # ----------------------------------------------------------
                self._checkpoint_db_experiences(step=step, epoch=epoch, batch=batch_idx)

                checkpoint_every = self.config.practice.checkpoint_every
                if (
                    checkpoint_every
                    and new_batches_this_session % checkpoint_every == 0
                    and new_batches_this_session > 0
                ):
                    next_step = step + 1
                    _tqdm_write(
                        f"  [step {step}] Checkpoint reached after "
                        f"{new_batches_this_session} new batch(es). "
                        f"Stopping; resume from step {next_step}."
                    )
                    raise TrainingCheckpoint(
                        next_step=next_step, epoch=epoch, batch=batch_idx
                    )

                # ----------------------------------------------------------
                # 4. Optional evaluation
                # ----------------------------------------------------------
                if self.config.practice.do_eval and self._should_evaluate(
                    step=step, batch_idx=batch_idx, num_batches=num_batches
                ):
                    _tqdm_write(f"  [step {step}] Running evaluation…")
                    await self._run_eval_ablation_per_db(epoch=epoch, step=step)

            batch_bar.close()

        # ------------------------------------------------------------------
        # Final output: write one experienced-prompt file per db.
        # ------------------------------------------------------------------
        self.db_experienced_prompt_paths: dict[str, Path] = {}
        if self._db_experiences:
            self.db_experienced_prompt_paths = self._write_db_experienced_prompts(
                self._db_experiences
            )

        return self._db_experiences

    # ------------------------------------------------------------------
    # Evaluation adapted for db-specific experience injection
    # ------------------------------------------------------------------

    async def _run_eval_ablation_per_db(self, *, epoch: int, step: int) -> None:
        """Evaluation pass using per-db experience injection."""
        assert self.config.data is not None
        from ..data_manager import JsonlDataManager

        eval_path = self.config.data.eval_path or self.config.data.practice_path
        truncate = self.config.practice.eval_data_truncate
        concurrency = self.config.evaluation.concurrency

        eval_data_manager = JsonlDataManager.from_path(eval_path, self.config)
        samples = eval_data_manager.load_base_samples(truncate=truncate)
        if not samples:
            logger.warning(
                "Eval dataset is empty; skipping evaluation at step %d.", step
            )
            return

        logger.info(
            "Running db-specific eval at epoch=%d step=%d over %d samples.",
            epoch, step, len(samples),
        )

        sem = asyncio.Semaphore(concurrency)
        task_timeout = self.config.practice.task_timeout
        rewards: list[float] = []
        failure_count: int = 0

        async def _eval_one(s: PracticeSample) -> None:
            nonlocal failure_count
            async with sem:
                try:
                    sample_db_path = self._rollout_manager._resolve_db_path(s)
                    db_exp_text = self._get_experiences_text(_db_key(s))
                    agent_out = await self._run_agent_once(
                        sample_db_path, s.question, db_exp_text,
                        task_timeout=task_timeout,
                    )
                    final_answer = str(agent_out.get("final_answer", "")) or ""
                    vr = self._verify_func(
                        {
                            "id": s.id,
                            "question": s.question,
                            "answer": s.answer,
                            "final_answer": final_answer,
                            "output_csv_path": agent_out.get("output_csv_path"),
                            "trajectory": list(agent_out.get("trajectory") or []),
                            "token_usage": dict(agent_out.get("token_usage") or {}),
                            "meta": s.meta,
                        }
                    )
                    rewards.append(float(vr.get("reward", 0.0)))
                except Exception:
                    logger.exception(
                        "Eval rollout failed for sample %s at step %d.", s.id, step
                    )
                    failure_count += 1
                    rewards.append(0.0)

        await asyncio.gather(*[_eval_one(s) for s in samples])

        if failure_count:
            logger.warning(
                "Eval step %d: %d/%d rollouts failed.", step, failure_count, len(samples)
            )

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        stat_key = f"eval_epoch{epoch}_step{step}"
        self.recorder.stat_update(
            {
                stat_key: {
                    "epoch": epoch,
                    "step": step,
                    "num_samples": len(samples),
                    "mean_reward": mean_reward,
                }
            }
        )
        logger.info(
            "Eval complete: epoch=%d step=%d mean_reward=%.4f", epoch, step, mean_reward
        )

    # ------------------------------------------------------------------
    # Rollout reconstruction helper (shared with parent cache-replay path)
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_rollouts(cached_rollout_dicts: list[dict]) -> list[RolloutResult]:
        """Re-hydrate RolloutResult objects from cached plain dicts."""
        from ..data_manager import PracticeSample as _PS

        rollouts = []
        for r in cached_rollout_dicts:
            s_dict = r["sample"]
            sample = _PS(
                id=s_dict["id"],
                question=s_dict["question"],
                answer=s_dict.get("answer"),
                meta=s_dict.get("meta") or {},
                stage=s_dict.get("stage", "init"),
                trajectories=s_dict.get("trajectories"),
                reward=s_dict.get("reward"),
                reasoning=s_dict.get("reasoning"),
                plan_hint=s_dict.get("plan_hint"),
                plan_axes=s_dict.get("plan_axes"),
            )
            rollouts.append(
                RolloutResult(
                    sample=sample,
                    final_answer=r.get("final_answer", ""),
                    trajectory=r.get("trajectory") or [],
                    reward=r.get("reward") or 0.0,
                    reasoning=r.get("reasoning"),
                    llmop_tokens=r.get("llmop_tokens", 0),
                    num_steps=r.get("num_steps", 0),
                    token_usage=r.get("token_usage"),
                    output_csv_path=r.get("output_csv_path"),
                    exec_profile=r.get("exec_profile") or None,
                    plan_hint=r.get("plan_hint"),
                    plan_axes=r.get("plan_axes"),
                    timed_out=r.get("timed_out", False),
                )
            )
        return rollouts
