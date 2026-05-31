"""CLI entrypoint for running training-free GRPO with the DB agent."""

from __future__ import annotations
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
load_dotenv()  # Load .env before any config or model initialisation

from .logging_setup import setup_run_logging
from .enumgrpo import TrainingFreeGRPO, TrainingCheckpoint
from .utils import parse_practice_config

# Name of the checkpoint state file written under the experiment directory.
# Stored one level above the per-run timestamped directory so it survives
# across runs of the same exp_id and can be found on restart.
_CHECKPOINT_FILE = "checkpoint.json"


def _exp_dir(run_log_dir: Path) -> Path:
    """Return the experiment-level directory (parent of the timestamped run dir)."""
    return run_log_dir.parent


def _find_checkpoint_step(run_log_dir: Path | None) -> int | None:
    """Return the resume_from_step stored in a previous checkpoint, or None."""
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
    """Persist checkpoint metadata at the experiment level and return the file path."""
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
    run_log_dir = setup_run_logging(cfg)

    # Auto-resume: if restart_step is not manually set in config but a
    # checkpoint.json exists from a previous run in the same log dir, use it.
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
        tf = TrainingFreeGRPO(cfg, run_log_dir=run_log_dir)
        experiences = asyncio.run(tf.run())

    except TrainingCheckpoint as ckpt:
        # Clean stop at a checkpoint interval — not an error.
        if run_log_dir is not None:
            ckpt_file = _write_checkpoint(run_log_dir, ckpt)
            print(
                f"\n=== Checkpoint reached (step {ckpt.next_step - 1} completed) ===\n"
                f"State saved to: {ckpt_file}\n"
                f"\nTo resume, re-run the same command:\n"
                f"  python -m learning.cli --config learning/config.yaml\n"
                f"  (restart_step will be set automatically from checkpoint.json)\n"
                f"\nOr to resume from a specific step:\n"
                f"  python -m learning.cli --config learning/config.yaml "
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
        sys.__stderr__.flush()
        os._exit(0)

    except Exception:
        # Print the traceback to the real terminal so it's visible even though
        # sys.stderr has been redirected to the log file.
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        os._exit(1)

    print("\n=== Training-free GRPO Experiences ===", file=sys.__stderr__)
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
        # If we completed fully, clear any stale checkpoint file so a re-run
        # starts fresh rather than re-playing an old resume_from_step.
        ckpt_file = _exp_dir(run_log_dir) / _CHECKPOINT_FILE
        if ckpt_file.exists():
            ckpt_file.unlink(missing_ok=True)

    # All data is on disk; force-exit without waiting for thread-pool shutdown.
    # sys.exit(0) would block here because asyncio.to_thread() leaves non-daemon
    # worker threads alive until the ThreadPoolExecutor is garbage-collected.
    sys.__stderr__.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
