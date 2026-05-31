"""Utility helpers for the training-free GRPO implementation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .config import DataConfig, PracticeConfig, load_config


@dataclass
class TaskRecorder:
    """In-memory record of a training-free GRPO run."""

    experiment_name: str | None = None
    experiences: dict[str, str] | None = None
    stats: dict[str, Any] | None = None

    def experiences_update(self, experiences: dict[str, str]) -> None:
        self.experiences = experiences

    def stat_update(self, stat: dict[str, Any]) -> None:
        if self.stats is None:
            self.stats = {}
        self.stats.update(stat)


def parse_practice_config(argv: list[str] | None = None) -> PracticeConfig:
    """Parse CLI arguments into a PracticeConfig.

    If ``--config`` is provided, its YAML values are used as the base layer.
    Every explicit CLI flag then overrides the corresponding YAML value.
    If ``--config`` is omitted, ``learning/config.yaml`` supplies the defaults.
    """
    parser = argparse.ArgumentParser(
        description="Run training-free GRPO experience generation for the DB agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config file (default: learning/config.yaml). "
            "Values in the file are used as defaults; CLI flags override them."
        ),
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Experiment identifier (used for logging and caching).",
    )
    parser.add_argument(
        "--db_files_dir",
        type=str,
        default=None,
        help=(
            "Directory containing DuckDB files.  When each sample has a 'db' field "
            "(e.g. 'california_schools'), the file is resolved as "
            "<db_files_dir>/<db>.duckdb.  Overrides the DB_FILES_DIR env var."
        ),
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Path to the agent system prompt file.  Overrides the default db_system_prompt.md.",
    )

    # Data arguments
    parser.add_argument(
        "--practice_path",
        type=str,
        default=None,
        help="Path to JSONL file with practice samples.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="Optional JSONL file for evaluation; if omitted, practice data is reused.",
    )

    # Practice arguments
    parser.add_argument("--epochs", type=int, default=None, help="Number of practice epochs.")
    parser.add_argument("--queries_per_update", type=int, default=None, help="Number of queries to roll out before running distillation once.")
    parser.add_argument("--grpo_n", type=int, default=None, help="Number of rollouts per query (group size).")
    parser.add_argument(
        "--rollout_concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent rollouts (agent runs).",
    )
    parser.add_argument(
        "--restart_step",
        type=int,
        default=None,
        help="Step index to restart from; if None, reuse cached results for all steps when available.",
    )
    parser.add_argument(
        "--task_timeout",
        type=int,
        default=None,
        help="Per-rollout timeout in seconds before the agent call is cancelled and retried.",
    )
    parser.add_argument(
        "--max_steps_per_rollout",
        type=int,
        default=None,
        help="Maximum number of tool calls (iterations) per agent rollout. Defaults to 600.",
    )
    parser.add_argument(
        "--mcp_result_max_chars",
        type=int,
        default=None,
        help="Maximum chars for a single tool result returned to the agent (0 = disabled). Defaults to 4000.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Maximum retry attempts per rollout on timeout or error.",
    )

    # Experience arguments
    parser.add_argument(
        "--agent_objective",
        type=str,
        default=None,
        help="Short description of the DB agent's objective (input / output).",
    )
    parser.add_argument(
        "--learning_objective",
        type=str,
        default=None,
        help="Short description of what the practice loop should help the agent learn.",
    )
    parser.add_argument(
        "--num_experiences_per_query",
        type=int,
        default=None,
        help="Number of experiences to extract per query.",
    )
    parser.add_argument(
        "--distillation_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent LLM calls during experience distillation.",
    )

    # Eval arguments (lightweight)
    parser.add_argument(
        "--do_eval",
        action="store_true",
        default=None,
        help="If set, run periodic evaluation using eval_path (or practice_path when omitted).",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=["epoch", "steps"],
        default=None,
        help="When to run evaluation if enabled.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluate every N steps when eval_strategy='steps'.",
    )
    parser.add_argument(
        "--eval_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent agent runs during evaluation.",
    )
    parser.add_argument(
        "--verify_module",
        type=str,
        default=None,
        help="Python module path for a verification function (e.g. 'learning.verify').",
    )
    parser.add_argument(
        "--verify_func_name",
        type=str,
        default=None,
        help="Function name inside verify_module implementing the verification logic.",
    )

    # Reward weight overrides
    parser.add_argument("--reward_w_sr", type=float, default=None, help="Reward weight for success rate.")
    parser.add_argument("--reward_w_row", type=float, default=None, help="Reward weight for row-level F1.")
    parser.add_argument("--reward_w_item", type=float, default=None, help="Reward weight for item-level F1.")
    parser.add_argument("--reward_w_cost", type=float, default=None, help="Reward weight for composite cost penalty (applied group-relative).")
    parser.add_argument("--reward_w_token", type=float, default=None, help="Weight of LLM-op token rank within the composite cost (w_token + w_steps should sum to 1).")
    parser.add_argument("--reward_w_steps", type=float, default=None, help="Weight of MCP step-count rank within the composite cost (w_token + w_steps should sum to 1).")

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-run log files.  Each run writes to "
            "<log_dir>/<exp_id>/<YYYYMMDD_HHMMSS>/.  "
            "Pass an empty string to disable file logging."
        ),
    )

    args = parser.parse_args(argv)

    # --- Load YAML base layer ---
    cfg = load_config(args.config)
    p = cfg.practice
    e = cfg.evaluation

    # --- Override with any CLI values that were explicitly provided ---
    if args.exp_id is not None:
        cfg.exp_id = args.exp_id
        e.exp_id = args.exp_id
    if args.db_files_dir is not None:
        cfg.db_files_dir = args.db_files_dir or None
    if args.system_prompt_path is not None:
        cfg.system_prompt_path = args.system_prompt_path or None
    if args.log_dir is not None:
        # empty string disables file logging; any non-empty string sets the dir
        cfg.log_dir = args.log_dir or None

    # data
    practice_path = args.practice_path or (str(cfg.data.practice_path) if cfg.data else None)
    eval_path = args.eval_path or (str(cfg.data.eval_path) if cfg.data and cfg.data.eval_path else None)
    if practice_path:
        cfg.data = DataConfig(
            practice_path=Path(practice_path),
            eval_path=Path(eval_path) if eval_path else None,
        )

    # practice
    if args.epochs is not None:
        p.epochs = args.epochs
    if args.queries_per_update is not None:
        p.queries_per_update = args.queries_per_update
    if args.grpo_n is not None:
        p.grpo_n = args.grpo_n
    if args.rollout_concurrency is not None:
        p.rollout_concurrency = args.rollout_concurrency
    if args.restart_step is not None:
        p.restart_step = args.restart_step
    if args.task_timeout is not None:
        p.task_timeout = args.task_timeout
    if args.max_steps_per_rollout is not None:
        p.max_steps_per_rollout = args.max_steps_per_rollout
    if args.mcp_result_max_chars is not None:
        p.mcp_result_max_chars = args.mcp_result_max_chars
    if args.max_retries is not None:
        p.max_retries = args.max_retries
    if args.agent_objective is not None:
        p.agent_objective = args.agent_objective
    if args.learning_objective is not None:
        p.learning_objective = args.learning_objective
    if args.num_experiences_per_query is not None:
        p.num_experiences_per_query = args.num_experiences_per_query
    if args.distillation_concurrency is not None:
        p.distillation_concurrency = args.distillation_concurrency
    if args.do_eval:
        p.do_eval = True
    if args.eval_strategy is not None:
        p.eval_strategy = args.eval_strategy
    if args.eval_steps is not None:
        p.eval_steps = args.eval_steps

    # evaluation
    if args.eval_concurrency is not None:
        e.concurrency = args.eval_concurrency
    if args.verify_module is not None:
        e.verify_module = args.verify_module
    if args.verify_func_name is not None:
        e.verify_func_name = args.verify_func_name
    if args.reward_w_sr is not None:
        e.reward_w_sr = args.reward_w_sr
    if args.reward_w_row is not None:
        e.reward_w_row = args.reward_w_row
    if args.reward_w_item is not None:
        e.reward_w_item = args.reward_w_item
    if args.reward_w_cost is not None:
        e.reward_w_cost = args.reward_w_cost
    if args.reward_w_token is not None:
        e.reward_w_token = args.reward_w_token
    if args.reward_w_steps is not None:
        e.reward_w_steps = args.reward_w_steps

    if not cfg.db_files_dir:
        parser.error(
            "--db_files_dir is required (or set DB_FILES_DIR in .env)."
        )
    if cfg.data is None:
        parser.error("--practice_path is required (or set data.practice_path in the config YAML).")

    return cfg


# ---------------------------------------------------------------------------
# Progress bar helpers
# All bars write to sys.__stderr__ (the original terminal fd) so they are
# visible even after sys.stderr has been redirected to a log file.
#
# Layout (four levels, all left-aligned):
#   Epoch   0/ 1  ████████████████████  100%  [00:02]
#     Batch   2/ 5  ████████░░░░░░░░░░░░   40%  [00:35]  step=9
#       Rollouts   3/15  ████░░░░░░░░░░░░░░░░   20%  [00:45]  reward=0.512  done=3
#         ▸ california_1        #3 run_sql
#         ▸ superhero_2         #1 open_session
# ---------------------------------------------------------------------------

_BAR_W = 20  # width of the ASCII progress bar block

EPOCH_FMT = (
    "Epoch  {{n:>{w}}}/{{total:<{w}}}  {{bar:{b}}}  {{percentage:3.0f}}%  [{{elapsed}}]"
).format(w=2, b=_BAR_W)

BATCH_FMT = (
    "  Batch  {{n:>{w}}}/{{total:<{w}}}  {{bar:{b}}}  {{percentage:3.0f}}%  [{{elapsed}}]  {{postfix}}"
).format(w=2, b=_BAR_W)

ROLLOUT_FMT = (
    "    Rollouts  {{n:>{w}}}/{{total:<{w}}}  {{bar:{b}}}  {{percentage:3.0f}}%  [{{elapsed}}]  {{postfix}}"
).format(w=3, b=_BAR_W)

DISTILL_FMT = (
    "    {{desc:<18}}  {{n:>{w}}}/{{total:<{w}}}  {{bar:{b}}}  {{percentage:3.0f}}%  [{{elapsed}}]"
).format(w=3, b=_BAR_W)

STATUS_FMT = "      ▸ {desc:<20} {postfix}"
STATUS_DESC_W = 20


def epoch_bar(total: int) -> tqdm:
    return tqdm(range(total), bar_format=EPOCH_FMT, file=sys.__stderr__)


def batch_bar(total: int, epoch: int) -> tqdm:  # noqa: ARG001
    return tqdm(range(total), bar_format=BATCH_FMT, leave=False, file=sys.__stderr__)


def rollout_bar(total: int) -> tqdm:
    return tqdm(total=total, bar_format=ROLLOUT_FMT, leave=False, file=sys.__stderr__)


def distill_bar(total: int, desc: str) -> tqdm:
    """Progress bar for one distillation stage (summarise / advantage / update)."""
    return tqdm(total=total, desc=desc, bar_format=DISTILL_FMT, leave=False, file=sys.__stderr__)


def status_bar(query_id: str, position: int) -> tqdm:
    """Single-line status bar for one active rollout (no progress block)."""
    return tqdm(
        total=0,
        desc=query_id[:STATUS_DESC_W],
        bar_format=STATUS_FMT,
        position=position,
        leave=False,
        file=sys.__stderr__,
        dynamic_ncols=False,
    )


def tqdm_write(msg: str) -> None:
    """Write a message to the terminal without disrupting progress bars."""
    tqdm.write(msg, file=sys.__stderr__)


