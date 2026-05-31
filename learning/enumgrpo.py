"""High-level orchestrator for training-free GRPO in the DB agent project."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from agent.codebase.runtime import DbAgentRuntime
from agent.codebase.config import RuntimeConfig, get_api_key, get_base_url, get_model_name

from .config import PracticeConfig
from .data_manager import JsonlDataManager, PracticeSample
from .experience_updater import ExperienceUpdater
from .rollout_manager import RolloutManager, RolloutResult
from .utils import TaskRecorder, epoch_bar as _epoch_bar, batch_bar as _batch_bar, tqdm_write as _tqdm_write

logger = logging.getLogger(__name__)


class TrainingCheckpoint(Exception):
    """Raised by the training loop when a checkpoint interval is reached.

    Carries the *next* step index to resume from so the CLI can write a
    ``checkpoint.json`` and exit cleanly.  This is intentionally an exception
    (not a return value) so it unwinds the nested loop without extra flags.
    """

    def __init__(self, next_step: int, epoch: int, batch: int) -> None:
        super().__init__(f"Checkpoint: completed up to step {next_step - 1}; resume from step {next_step}")
        self.next_step = next_step
        self.epoch = epoch
        self.batch = batch


class TrainingFreeGRPO:
    """Run training-free GRPO around the DB agent.

    This class mirrors the high-level structure of `utu.practice.TrainingFreeGRPO`
    but uses the `DbAgentRuntime` as the underlying policy and a lightweight,
    file-based dataset / rollout implementation.
    """

    def __init__(self, config: PracticeConfig, *, run_log_dir: Path | None = None) -> None:
        if config.data is None:
            raise ValueError("PracticeConfig.data must be set.")
        self.config = config
        # When a training log directory is provided, agent run dirs are placed
        # under <run_log_dir>/rollouts/ so all artefacts for a run stay together.
        self._run_log_dir: Path | None = run_log_dir
        self.recorder = TaskRecorder(experiment_name=config.exp_id)
        self._experience_updater = ExperienceUpdater(config)
        self._verify_func: Callable[[dict], dict] = self._load_verify_func()

    async def run(self) -> dict[str, str]:
        """Run the complete experience generation process.

        Writes a single ``experienced_prompt.md`` file to the run log dir and
        stores its path in ``self.experienced_prompt_path``.

        Returns:
            dict: Mapping from experience ID (e.g. 'G0') to textual guideline.
        """
        self._rollout_manager = RolloutManager(
            self.config,
            run_agent_once=self._run_agent_once,
            verify_func=self._verify_func,
        )
        rollout_manager = self._rollout_manager

        # Main epoch / batch loop.
        epoch_bar = _epoch_bar(self.config.practice.epochs)
        #
        # Pre-seed the recorder with the latest cached experience pool so that
        # if the per-step replay below is skipped (e.g. a rollout cache entry is
        # missing for some middle step) the recorder still starts from a valid
        # pool rather than from scratch.
        restart_step = self.config.practice.restart_step
        if restart_step:
            for seed_step in range(restart_step - 1, -1, -1):
                seed_exps = self._load_experiences(seed_step)
                if seed_exps is not None:
                    self.recorder.experiences_update(seed_exps)
                    _tqdm_write(
                        f"  [resume] Pre-seeded experience pool from step {seed_step} "
                        f"({len(seed_exps)} experiences)."
                    )
                    break

        # Count of batches for which new work (rollouts + distillation) was done
        # in this session.  Cache-replayed batches do NOT count toward the
        # checkpoint interval so that resuming from a checkpoint doesn't
        # immediately re-trigger the stop condition.
        new_batches_this_session = 0
        for epoch in epoch_bar:
            epoch_data = await rollout_manager.load_epoch_data_async(epoch)
            total = len(epoch_data)
            # Cap queries_per_update to the actual dataset size so a large
            # configured value still works on small datasets.
            queries_per_update = min(self.config.practice.queries_per_update, total // self.config.practice.grpo_n)
            if queries_per_update < 1:
                raise ValueError(
                    f"Epoch {epoch}: dataset too small ({total} expanded samples) for "
                    f"grpo_n={self.config.practice.grpo_n} — need at least {self.config.practice.grpo_n} samples."
                )
            group_size = queries_per_update * self.config.practice.grpo_n
            num_batches = math.ceil(total / group_size)

            batch_bar = _batch_bar(num_batches, epoch)
            for batch_idx in batch_bar:
                step = epoch * num_batches + batch_idx
                batch_bar.set_postfix_str(f"step={step}")

                # Check rollout cache first; run agent only on a cache miss.
                cached_rollout_dicts = self._load_rollouts(step)
                if cached_rollout_dicts is not None and self._should_use_cache(step):
                    rollouts = []
                    for r in cached_rollout_dicts:
                        s_dict = r["sample"]
                        sample = PracticeSample(
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
                        rollouts.append(RolloutResult(
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
                        ))
                    _tqdm_write(f"  [step {step}] Loaded {len(rollouts)} rollouts from cache.")
                else:
                    rollouts = await rollout_manager.run_batch(
                        epoch_samples=epoch_data,
                        batch_idx=batch_idx,
                        queries_per_update=queries_per_update,
                        recorder=self.recorder,
                        desc=f"  E{epoch} B{batch_idx} rollouts",
                    )
                    # Persist rollouts so a crash/restart can skip re-running the agent.
                    if rollouts and self._should_use_cache(step):
                        self._save_rollouts(step, rollouts)

                if not rollouts:
                    continue

                rewards = [r.reward for r in rollouts]
                mean_r = sum(rewards) / len(rewards) if rewards else 0.0
                batch_bar.set_postfix(step=step, mean_reward=f"{mean_r:.3f}")

                # 1) Try to reuse cached experiences for this step, if allowed.
                cached = self._load_experiences(step)
                if cached is not None and self._should_use_cache(step):
                    self.recorder.experiences_update(cached)
                    _tqdm_write(f"  [step {step}] Loaded {len(cached)} experiences from run dir.")
                else:
                    # 2) Update experiences based on group-relative performance.
                    _tqdm_write(f"  [step {step}] Distilling experiences from {len(rollouts)} rollouts "
                               f"(mean_reward={mean_r:.3f})…")
                    new_exps = await self._experience_updater.run(
                        rollouts=rollouts,
                        recorder=self.recorder,
                        log_dir=self._run_log_dir,
                        step=step,
                        epoch=epoch,
                        batch=batch_idx,
                    )
                    _tqdm_write(f"  [step {step}] Distilled {len(new_exps)} experiences.")
                    new_batches_this_session += 1

                # 4a) Write an incremental experience checkpoint to the run log dir.
                self._checkpoint_experiences(step=step, epoch=epoch, batch=batch_idx)

                # 4b) If checkpoint_every is set, stop after every K *newly computed*
                # batches in this session.  Cache-replayed batches are skipped so
                # resuming from a checkpoint never immediately re-triggers the stop.
                checkpoint_every = self.config.practice.checkpoint_every
                if checkpoint_every and new_batches_this_session % checkpoint_every == 0 and new_batches_this_session > 0:
                    next_step = step + 1
                    _tqdm_write(
                        f"  [step {step}] Checkpoint reached after {new_batches_this_session} "
                        f"new batch(es) (checkpoint_every={checkpoint_every}). "
                        f"Stopping; resume from step {next_step}."
                    )
                    raise TrainingCheckpoint(next_step=next_step, epoch=epoch, batch=batch_idx)

                # 4c) Run evaluation if scheduled.
                if self.config.practice.do_eval and self._should_evaluate(
                    step=step, batch_idx=batch_idx, num_batches=num_batches
                ):
                    _tqdm_write(f"  [step {step}] Running evaluation…")
                    await self._run_eval(epoch=epoch, step=step)

            batch_bar.close()

        final_experiences = self.recorder.experiences or {}
        self.experienced_prompt_path: Path | None = None
        if final_experiences:
            self.experienced_prompt_path = self._write_experienced_prompt(final_experiences)
        return final_experiences

    def _checkpoint_experiences(self, *, step: int, epoch: int, batch: int) -> None:
        """Write the current experience pool as a checkpoint.

        Writes to two locations:
        - ``<run_log_dir>/experiences/`` — per-run historical copy for inspection.
        - ``<exp_dir>/experiences/``     — experiment-level canonical copy, read
          on resume so that the correct step's experiences are always found
          regardless of which timestamped run dir they were produced in.

        Does nothing when no run log dir has been configured or when the
        experience pool is empty.
        """
        if self._run_log_dir is None or not self.recorder.experiences:
            return

        exps = self.recorder.experiences
        payload = {
            "step": step,
            "epoch": epoch,
            "batch": batch,
            "num_experiences": len(exps),
            "experiences": exps,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)

        # Per-run copy (historical, one per timestamped run dir).
        run_ckpt_dir = self._run_log_dir / "experiences"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        (run_ckpt_dir / f"step_{step:04d}.json").write_text(text, encoding="utf-8")
        (run_ckpt_dir / "latest.json").write_text(text, encoding="utf-8")

        # Experiment-level copy (canonical for resume across runs).
        exp_ckpt_dir = self._run_log_dir.parent / "experiences"
        exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        (exp_ckpt_dir / f"step_{step:04d}.json").write_text(text, encoding="utf-8")
        (exp_ckpt_dir / "latest.json").write_text(text, encoding="utf-8")

        logger.debug("Wrote experience checkpoint for step %d", step)

    def _write_experienced_prompt(self, experiences: dict[str, str]) -> Path | None:
        """Assemble and write the final system prompt with learned experiences.

        Writes a single file to ``<run_log_dir>/experienced_prompt.md``.
        Returns the path, or ``None`` when no run log dir is configured.
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

        base_text = base_prompt_path.read_text(encoding="utf-8") if base_prompt_path.exists() else ""

        exp_lines = [
            "\n\n## Learned Experiences (Training-Free GRPO)",
            "",
            "When solving problems, you MUST first carefully read and understand "
            "the helpful instructions and experiences:",
        ]
        for k, v in sorted(experiences.items()):
            exp_lines.append(f"[{k}]. {v}")

        content = base_text.rstrip() + "\n" + "\n".join(exp_lines) + "\n"

        if self._run_log_dir is None:
            logger.warning("No run_log_dir configured; experienced prompt was not written.")
            return None

        out_path = self._run_log_dir / "experienced_prompt.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Wrote experienced prompt to %s", out_path)
        return out_path

    def _should_use_cache(self, step: int) -> bool:
        restart_step = self.config.practice.restart_step
        return restart_step is None or step < restart_step

    # ------------------------------------------------------------------
    # Run-dir-based rollout / experience I/O
    # (replaces learning/.cache/ so everything lives under exp/<exp_id>/)
    # ------------------------------------------------------------------

    def _rollout_cache_path(self, step: int) -> Path | None:
        """Path for the rollout cache file at the experiment level."""
        if self._run_log_dir is None:
            return None
        cache_dir = self._run_log_dir.parent / "rollout_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"step_{step:04d}.json"

    def _experience_cache_path(self, step: int) -> Path | None:
        """Path for the experiment-level experience file written by _checkpoint_experiences."""
        if self._run_log_dir is None:
            return None
        return self._run_log_dir.parent / "experiences" / f"step_{step:04d}.json"

    def _load_rollouts(self, step: int) -> list | None:
        path = self._rollout_cache_path(step)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupt rollout cache at %s; will re-run step.", path)
            return None
        if not isinstance(raw, list):
            logger.warning("Invalid rollout cache at %s (expected list); will re-run step.", path)
            return None
        return raw

    def _save_rollouts(self, step: int, rollouts: list) -> None:
        path = self._rollout_cache_path(step)
        if path is None:
            return
        from dataclasses import asdict
        path.write_text(
            json.dumps([asdict(r) for r in rollouts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Saved rollout cache for step %d to %s", step, path)

    def _load_experiences(self, step: int) -> dict | None:
        path = self._experience_cache_path(step)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupt experience file at %s; will re-run step.", path)
            return None
        exps = raw.get("experiences")
        if not isinstance(exps, dict):
            logger.warning("Invalid experience file at %s (unexpected format); will re-run step.", path)
            return None
        return exps

    def _should_evaluate(self, *, step: int, batch_idx: int, num_batches: int) -> bool:
        strategy = self.config.practice.eval_strategy
        if strategy == "epoch":
            # Evaluate at the last batch of each epoch.
            return batch_idx == num_batches - 1
        # strategy == "steps"
        return (step + 1) % self.config.practice.eval_steps == 0

    async def _run_eval(self, *, epoch: int, step: int) -> None:
        """Run a single evaluation pass over the eval dataset.

        Loads eval samples (no grpo_n duplication), runs one agent call per
        question with the current experience pool, scores each via verify_func,
        and records aggregate stats in recorder.stats.
        """
        assert self.config.data is not None
        eval_path = self.config.data.eval_path or self.config.data.practice_path
        truncate = self.config.practice.eval_data_truncate
        concurrency = self.config.evaluation.concurrency

        # Load base eval samples directly (1 attempt per question).
        eval_data_manager = JsonlDataManager.from_path(eval_path, self.config)
        samples = eval_data_manager.load_base_samples(truncate=truncate)
        if not samples:
            logger.warning("Eval dataset is empty; skipping evaluation at step %d.", step)
            return

        logger.info(
            "Running eval at epoch=%d step=%d over %d samples (concurrency=%d).",
            epoch, step, len(samples), concurrency,
        )

        experiences_text: str | None = None
        if self.recorder.experiences:
            experiences_text = "\n".join(f"[{k}]. {v}" for k, v in self.recorder.experiences.items())

        sem = asyncio.Semaphore(concurrency)
        task_timeout = self.config.practice.task_timeout
        rewards: list[float] = []
        failure_count: int = 0

        async def _eval_one(s: PracticeSample) -> None:
            nonlocal failure_count
            async with sem:
                try:
                    sample_db_path = self._rollout_manager._resolve_db_path(s)
                    agent_out = await self._run_agent_once(
                        sample_db_path, s.question, experiences_text,
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
                    logger.exception("Eval rollout failed for sample %s at step %d.", s.id, step)
                    failure_count += 1
                    rewards.append(0.0)

        await asyncio.gather(*[_eval_one(s) for s in samples])

        if failure_count:
            logger.warning(
                "Eval step %d: %d/%d rollouts failed (reward set to 0.0 for each); "
                "mean_reward may be underestimated.",
                step, failure_count, len(samples),
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
        logger.info("Eval complete: epoch=%d step=%d mean_reward=%.4f", epoch, step, mean_reward)

    _SQL_PREVIEW_LEN = 120   # chars of SQL shown in exec profile
    _CELL_PREVIEW_LEN = 40  # chars per result cell
    _RESULT_PREVIEW_COLS = 5  # max columns shown in result preview
    _RESULT_PREVIEW_ROWS = 2  # max data rows shown in result preview

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = str(s)
        return (s[:n] + "…") if len(s) > n else s

    def _build_exec_profile(self, run_dir: Path) -> list[dict]:
        """Read the MCP server JSONL logs and build a compact per-step execution profile.

        Each entry covers one MCP tool call and contains fields useful for
        experience distillation:
            tool              — tool name (e.g. "run_sql", "llm_map")
            duration_ms       — actual wall-clock latency
            op_preview        — truncated SQL / relation name / question, depending on tool
            input_rows        — rows fed into the operation (LLM ops only)
            row_count         — rows produced / returned by the operation
            result_preview    — {"cols": [...], "rows": [[...]]} for data-returning SQL ops
            llm_input_tokens  — prompt tokens consumed (LLM ops only)
            llm_output_tokens — completion tokens generated (LLM ops only)
            error             — error message if the call failed (None on success)
        """
        logs_dir = run_dir / "logs" / "mcp_server"
        if not logs_dir.exists():
            return []

        profile: list[dict] = []
        try:
            for log_path in sorted(logs_dir.glob("*.jsonl")):
                try:
                    with log_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = (line or "").strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(ev, dict):
                                continue
                            tool = str(ev.get("tool") or "")
                            if not tool or tool in ("open_session", "close_session", "list_sessions"):
                                continue

                            args = ev.get("arguments") or {}

                            step: dict = {
                                "tool": tool,
                                "duration_ms": ev.get("duration_ms"),
                                "op_preview": None,
                                "input_rows": None,
                                "row_count": None,
                                "result_preview": None,
                                "llm_input_tokens": None,
                                "llm_output_tokens": None,
                                "error": ev.get("error") or None,
                            }

                            # --- Operation context (what was asked / queried) ---
                            if tool in ("run_sql", "materialize_temp"):
                                sql = str(args.get("sql") or "").strip().replace("\n", " ")
                                step["op_preview"] = self._truncate(sql, self._SQL_PREVIEW_LEN)
                            elif tool in ("describe_relation", "preview_relation"):
                                rel = str(args.get("relation") or "")
                                if rel:
                                    step["op_preview"] = rel
                            elif tool == "llm_reduce":
                                q = str(args.get("question") or "").strip()
                                step["op_preview"] = self._truncate(q, self._SQL_PREVIEW_LEN)
                            elif tool == "llm_map":
                                col = str(args.get("column") or args.get("question") or "").strip()
                                step["op_preview"] = self._truncate(col, self._SQL_PREVIEW_LEN)

                            result = ev.get("result")
                            if isinstance(result, dict):
                                if tool == "llm_map":
                                    lm = result.get("llm_map") or {}
                                    # token_usage_reports = number of distinct LLM calls
                                    # (may differ from input row count when distinct_from is used).
                                    reports = ev.get("llm_token_usage_reports") or lm.get("token_usage_reports")
                                    if reports is not None:
                                        try:
                                            step["input_rows"] = int(reports)
                                        except (ValueError, TypeError):
                                            pass
                                    rc = lm.get("row_count")
                                    if rc is not None:
                                        try:
                                            step["row_count"] = int(rc)
                                        except (ValueError, TypeError):
                                            pass

                                elif tool == "llm_reduce":
                                    ctx = result.get("context")
                                    if isinstance(ctx, dict):
                                        ctx_rc = ctx.get("row_count")
                                        if ctx_rc is not None:
                                            try:
                                                step["input_rows"] = int(ctx_rc)
                                            except (ValueError, TypeError):
                                                pass
                                    rc = result.get("row_count")
                                    if rc is not None:
                                        try:
                                            step["row_count"] = int(rc)
                                        except (ValueError, TypeError):
                                            pass

                                else:
                                    # SQL/schema ops: output row_count + result preview.
                                    rc = result.get("row_count")
                                    if rc is not None:
                                        try:
                                            step["row_count"] = int(rc)
                                        except (ValueError, TypeError):
                                            pass

                                    # Result preview: first few rows for data-returning ops.
                                    # Helps the distillation LLM see what the agent actually saw.
                                    if tool in ("run_sql", "preview_relation") and rc:
                                        cols = result.get("columns") or []
                                        rows = result.get("rows") or []
                                        if cols or rows:
                                            step["result_preview"] = {
                                                "cols": [str(c) for c in cols[:self._RESULT_PREVIEW_COLS]],
                                                "rows": [
                                                    [self._truncate(str(cell), self._CELL_PREVIEW_LEN)
                                                     for cell in row[:self._RESULT_PREVIEW_COLS]]
                                                    for row in rows[:self._RESULT_PREVIEW_ROWS]
                                                ],
                                            }

                            # Extract per-call LLM token breakdown (LLM ops only).
                            tu = ev.get("llm_token_usage")
                            if isinstance(tu, dict):
                                try:
                                    inp = tu.get("input_tokens")
                                    if inp is not None:
                                        step["llm_input_tokens"] = int(inp)
                                except (ValueError, TypeError):
                                    pass
                                try:
                                    out = tu.get("output_tokens")
                                    if out is not None:
                                        step["llm_output_tokens"] = int(out)
                                except (ValueError, TypeError):
                                    pass

                            profile.append(step)
                except Exception:
                    logger.debug(
                        "Failed to parse MCP log file %s; skipping for exec profile.",
                        log_path, exc_info=True,
                    )
                    continue
        except Exception:
            logger.warning(
                "exec-profile build failed entirely for run_dir %s; "
                "distillation will proceed without a profile.",
                run_dir, exc_info=True,
            )
        return profile

    async def _run_agent_once(self, db_path: str, question: str, experiences_text: str | None,
                              on_action=None, plan_hint: str | None = None,
                              task_timeout: float | None = None) -> dict:
        """Run the DB agent once on a given question and return structured output.

        This helper:
        - builds a transient DbAgentRuntime
        - points it at the provided DuckDB path
        - instructs the agent to write its result to a dedicated CSV file
        - sends a single message and returns the final answer, CSV path, and run metadata.

        If task_timeout is set and the agent exceeds it, the rollout is NOT
        treated as a failure.  Whatever partial trajectory has been written to
        disk is collected and returned with an empty final_answer so the batch
        can still use it for experience distillation.  The same applies when the
        agent hits max_steps_per_rollout: OpenHands exits the run loop normally
        (no exception), so partial results are already returned naturally.
        """
        # Resolve model and auth using the same helpers as the CLI.
        model = get_model_name(None)
        api_key = get_api_key()
        base_url = get_base_url()

        runtime_cfg = RuntimeConfig(model_name=model, api_key=api_key, base_url=base_url,
                                    temperature=self.config.practice.rollout_temperature,
                                    max_iteration_per_run=self.config.practice.max_steps_per_rollout,
                                    mcp_result_max_chars=self.config.practice.mcp_result_max_chars)

        from agent.codebase.config import PROJECT_ROOT

        # Use a UUID so concurrent rollouts in the same process never collide.
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_grpo"

        # Prefer placing rollout dirs under the training log directory so all
        # artefacts for a run are co-located.  Fall back to agent/runs/ when
        # no training log dir has been configured.
        if self._run_log_dir is not None:
            run_dir = self._run_log_dir / "rollouts" / run_id
        else:
            run_dir = Path(PROJECT_ROOT).resolve() / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated output CSV for this rollout — relative to the workspace so the
        # agent (which runs with workspace_dir as its CWD) can write to it without
        # needing an absolute path.
        workspace_dir = Path(PROJECT_ROOT).resolve().parent
        output_csv_path = run_dir / "output.csv"
        # Express as a path relative to workspace_dir for the prompt.
        try:
            output_csv_rel = output_csv_path.relative_to(workspace_dir)
        except ValueError:
            output_csv_rel = output_csv_path  # use absolute if not under workspace

        rt = DbAgentRuntime(workspace_dir=workspace_dir, run_dir=run_dir, runtime_cfg=runtime_cfg, auto_mode=True,
                            system_prompt_path=self.config.system_prompt_path or None,
                            db_files_dir=self.config.db_files_dir or None,
                            on_action=on_action)
        try:
            rt.set_db_context(db_path=db_path, read_only=True)
            # Build a short, focused prompt aimed at solving the question,
            # optionally enriched with current experiences.
            prompt_parts = [
                "You are running in training-free GRPO mode for a database agent.",
                f"DuckDB path: {db_path}",
                f"Output CSV path: {output_csv_rel}",
                "",
                "IMPORTANT: You MUST save your final query results as a CSV file to the path "
                f"specified above ({output_csv_rel}).  The file should include a header row "
                "followed by one data row per result.  This is required for automated evaluation.",
            ]
            if experiences_text:
                prompt_parts.append("")
                prompt_parts.append("Here are learned experiences you MUST follow when solving this task:")
                prompt_parts.append(experiences_text)
            if plan_hint:
                prompt_parts.append("")
                prompt_parts.append("Strategy Hint for this attempt:")
                prompt_parts.append(plan_hint)
                prompt_parts.append(
                    "(This hint describes a suggested structural approach. "
                    "Apply it when it fits the data; deviate only if the data clearly calls for a different approach.)"
                )
            prompt_parts.append("")
            prompt_parts.append("Task:")
            prompt_parts.append(question)
            prompt = "\n".join(prompt_parts).strip()
            timed_out = False
            try:
                if task_timeout is not None:
                    final_answer = await asyncio.wait_for(
                        asyncio.to_thread(rt.run, prompt),
                        timeout=task_timeout,
                    )
                else:
                    final_answer = await asyncio.to_thread(rt.run, prompt)
            except asyncio.TimeoutError:
                timed_out = True
                final_answer = ""
                logger.warning(
                    "Rollout timed out after %ds for run %s; collecting partial trajectory.",
                    task_timeout, run_id,
                )
            # Build a lightweight trajectory from the conversation event logs.
            traj: list[dict] = []
            try:
                events_root = run_dir / "logs" / "conversations" / "main_agent"
                if events_root.exists():
                    for conv_dir in sorted(events_root.glob("*")):
                        ev_dir = conv_dir / "events"
                        if not ev_dir.exists():
                            continue
                        for ev_path in sorted(ev_dir.glob("event-*.json")):
                            try:
                                raw = json.loads(ev_path.read_text(encoding="utf-8"))
                            except Exception:
                                logger.warning("Could not parse event file %s; skipping.", ev_path)
                                continue
                            traj_step = {
                                "id": raw.get("id"),
                                "timestamp": raw.get("timestamp"),
                                "source": raw.get("source"),
                                "kind": raw.get("kind"),
                                "summary": raw.get("summary"),
                                "tool_name": raw.get("tool_name"),
                                "action": raw.get("action"),
                                "thought": raw.get("thought"),
                                "reasoning_content": raw.get("reasoning_content"),
                            }
                            traj.append(traj_step)
            except Exception:
                logger.exception(
                    "Trajectory reconstruction failed for run %s; returning empty trajectory.", run_id
                )
                traj = []

            return {
                "final_answer":    final_answer,
                "output_csv_path": str(output_csv_path),
                "trajectory":      traj,
                "run_dir":         str(run_dir),
                "token_usage":     rt.get_llm_metrics(),
                "exec_profile":    self._build_exec_profile(run_dir),
                "timed_out":       timed_out,
            }
        finally:
            rt.close()

    def _load_verify_func(self) -> Callable[[dict], dict]:
        """Load the verification function specified in the config, if any."""
        mod_name = self.config.evaluation.verify_module
        func_name = self.config.evaluation.verify_func_name
        if not mod_name or not func_name:
            def _default_verify(sample: dict) -> dict:
                answer = str(sample.get("final_answer", "")).strip()
                return {"reward": 1.0 if answer else 0.0, "reasoning": None}
            return _default_verify

        module = __import__(mod_name, fromlist=[func_name])
        base_func = getattr(module, func_name)

        # Inject performance weights from config.  Cost (w_cost) is read by
        # experience_updater directly from EvalConfig and not used in verify_func.
        ec = self.config.evaluation
        weights = {
            "w_sr":  ec.reward_w_sr,
            "w_row": ec.reward_w_row,
            "w_item": ec.reward_w_item,
        }

        def _wrapped_verify(sample: dict) -> dict:
            s = dict(sample)
            s.setdefault("_weights", weights)
            return base_func(s)

        return _wrapped_verify
