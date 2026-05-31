"""CLI entrypoint for the independent-rollout distillation ablation.

This is a drop-in replacement for ``learning.cli`` that uses
:class:`~learning.ablation_ungrouped.grpo.TrainingFreeIndependent` instead
of :class:`~learning.enumgrpo.TrainingFreeGRPO`.  All flags, config
keys, checkpoint / resume logic, and logging behaviour are identical.

Run with:

    python -m learning.ablation_ungrouped.cli --config learning/config.yaml [overrides...]
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
load_dotenv()

from ..logging_setup import setup_run_logging
from ..enumgrpo import TrainingCheckpoint
from .grpo import TrainingFreeIndependent
from ..utils import parse_practice_config

_CHECKPOINT_FILE = "checkpoint.json"


def _exp_dir(run_log_dir: Path) -> Path:
    return run_log_dir.parent


def _find_checkpoint_step(run_log_dir: Path | None) -> int | None:
    if run_log_dir is None:
        return None
    ckpt = _exp_dir(run_log_dir) / _CHECKPOINT_FILE
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        step = data.get("resume_from_step")
        if isinstance(step, int) and step >= 0:
            return step
    except Exception:
        pass
    return None


def _write_checkpoint(run_log_dir: Path, ckpt: TrainingCheckpoint) -> Path:
    payload = {
        "resume_from_step": ckpt.next_step,
        "completed_epoch": ckpt.epoch,
        "completed_batch": ckpt.batch,
        "written_by_run": str(run_log_dir),
    }
    out = _exp_dir(run_log_dir) / _CHECKPOINT_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main() -> NoReturn:
    cfg = parse_practice_config()
    cfg.exp_id = cfg.exp_id + "_ablation_ungrouped"
    run_log_dir = setup_run_logging(cfg)

    if cfg.practice.restart_step is None and run_log_dir is not None:
        saved_step = _find_checkpoint_step(run_log_dir)
        if saved_step is not None:
            cfg.practice.restart_step = saved_step  # type: ignore[attr-defined]
            print(
                f"  [checkpoint] Auto-resuming from step {saved_step} "
                f"(loaded from {_exp_dir(run_log_dir) / _CHECKPOINT_FILE})",
                file=sys.__stderr__,
            )

    try:
        tf = TrainingFreeIndependent(cfg, run_log_dir=run_log_dir)
        experiences = asyncio.run(tf.run())

    except TrainingCheckpoint as ckpt:
        if run_log_dir is not None:
            ckpt_file = _write_checkpoint(run_log_dir, ckpt)
            print(
                f"\n=== Checkpoint reached (step {ckpt.next_step - 1} completed) ===\n"
                f"State saved to: {ckpt_file}\n"
                f"\nTo resume, re-run the same command:\n"
                f"  python -m learning.ablation_ungrouped.cli --config learning/config.yaml\n"
                f"  (restart_step will be set automatically from checkpoint.json)\n"
                f"\nOr to resume from a specific step:\n"
                f"  python -m learning.ablation_ungrouped.cli --config learning/config.yaml "
                f"--restart_step {ckpt.next_step}",
                file=sys.__stderr__,
            )
        else:
            print(
                f"\n=== Checkpoint reached (step {ckpt.next_step - 1} completed) ===\n"
                f"No run_log_dir configured; checkpoint state was NOT persisted.\n"
                f"To resume manually, pass --restart_step {ckpt.next_step}.",
                file=sys.__stderr__,
            )
        sys.exit(0)

    except Exception:
        traceback.print_exc(file=sys.__stderr__)
        sys.exit(1)

    print("\n=== Independent Rollout Distillation Experiences ===", file=sys.__stderr__)
    for k, v in sorted(experiences.items()):
        print(f"[{k}] {v}", file=sys.__stderr__)

    if tf.experienced_prompt_path is not None:
        print(f"\nExperienced prompt written to:\n  {tf.experienced_prompt_path}", file=sys.__stderr__)
        print(
            "\nTo use it with the DB agent CLI:\n"
            f"  python -m codebase db --system_prompt_path {tf.experienced_prompt_path} ...",
            file=sys.__stderr__,
        )
    else:
        print("\nNo experiences were produced; prompt file was not written.", file=sys.__stderr__)

    if run_log_dir is not None:
        print(f"\nRun logs saved to:\n  {run_log_dir}", file=sys.__stderr__)
        ckpt_file = _exp_dir(run_log_dir) / _CHECKPOINT_FILE
        if ckpt_file.exists():
            ckpt_file.unlink(missing_ok=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
