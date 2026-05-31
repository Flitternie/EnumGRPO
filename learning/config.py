"""Configuration models for training-free GRPO in the DB agent project.

This module intentionally mirrors the structure of `utu.config.practice_config`
but simplifies the dependencies so it can live entirely inside `db_revise`.

All hyperparameter defaults live in ``learning/config.yaml``.
Use ``load_config(path)`` to build a ``PracticeConfig`` from that file, then
override individual fields programmatically or via CLI flags.

These dataclasses are plain data containers — do not add default values here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(slots=True)
class PracticeArguments:
    """Arguments controlling the practice (training-free GRPO) loop."""

    # rollout
    epochs: int
    grpo_n: int
    queries_per_update: int
    rollout_concurrency: int
    rollout_temperature: float
    rollout_data_truncate: Optional[int]
    task_timeout: int
    max_retries: int
    shuffle_data: bool
    max_steps_per_rollout: Optional[int]   # cap tool calls per agent run (None = default 600)
    mcp_result_max_chars: Optional[int]    # cap tool result chars sent to agent (None = env/default 4000)

    # restart / caching
    restart_step: Optional[int]
    checkpoint_every: Optional[int]   # pause after every N batches; null = run straight through

    # experience update
    agent_objective: Optional[str]
    learning_objective: Optional[str]
    given_ground_truth: bool
    num_experiences_per_query: int
    distillation_concurrency: int

    # plan enumeration
    plan_library_path: Optional[str]    # path to plan_library.yaml; None = use bundled default
    plan_enumeration: bool              # enable/disable plan-enumeration rollout diversity

    # eval
    do_eval: bool
    eval_strategy: Literal["epoch", "steps"]
    eval_steps: int
    eval_data_truncate: Optional[int]


@dataclass(slots=True)
class DataConfig:
    """Where to load practice / eval data from."""

    # JSONL file containing practice samples:
    # {"id": str, "question": str, "answer": str | null, ...}
    practice_path: Path

    # Optional separate eval dataset; if unset, practice data is reused.
    eval_path: Optional[Path]


@dataclass(slots=True)
class EvalConfig:
    """Lightweight evaluation config for training-free GRPO."""

    exp_id: str
    concurrency: int
    pass_k: int

    # Verification function — must have signature:
    #   verify_func(sample: dict, timeout_score: float = 0, **kwargs) -> dict
    verify_module: Optional[str]
    verify_func_name: Optional[str]

    # Multi-objective reward weights (used by learning.verify and experience_updater).
    # Populated from learning/config.yaml via load_config(); injected into
    # every verify_func call — no env-var overrides needed.
    # cost is applied group-relative in experience_updater, not in verify_func.
    reward_w_sr: float
    reward_w_row: float
    reward_w_item: float
    reward_w_cost: float    # overall weight of the composite cost penalty
    reward_w_token: float   # weight of llmop token rank within the cost composite
    reward_w_steps: float   # weight of MCP step count rank within the cost composite


@dataclass(slots=True)
class PracticeConfig:
    """Unified configuration for training-free GRPO in this project."""

    exp_id: str
    practice: PracticeArguments
    data: Optional[DataConfig]
    evaluation: EvalConfig

    # Directory where per-run log files are written.
    log_dir: Optional[str]

    # Directory containing all DuckDB files.  When set, a bare db name in
    # sample metadata (e.g. "california_schools") is resolved as
    # <db_files_dir>/<name>.duckdb.  Required; set DB_FILES_DIR in .env.
    db_files_dir: Optional[str] = None

    # Path to the agent system prompt file.  When set, overrides the default
    # db_system_prompt.md used by DbAgentRuntime.
    system_prompt_path: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> PracticeConfig:
    """Load a ``PracticeConfig`` from a YAML file.

    If *path* is ``None``, the bundled ``learning/config.yaml`` is used.
    CLI flags returned by ``parse_practice_config`` always win over YAML values.

    The YAML schema mirrors the field layout of this module:

    .. code-block:: yaml

        experiment:
          exp_id: my_run
          db_path: /data/swan.duckdb
        data:
          practice_path: /data/train.jsonl
        practice:
          epochs: 5
          grpo_n: 8
        evaluation:
          reward_w_sr: 0.6

    Unknown keys are silently ignored.  All defaults are defined in the YAML —
    this function does not duplicate them.
    """
    import yaml  # soft dependency — only needed when loading YAML

    if path is None:
        path = Path(__file__).parent / "config.yaml"

    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}

    exp_sec: dict[str, Any] = raw.get("experiment", {})
    data_sec: dict[str, Any] = raw.get("data", {})
    prac_sec: dict[str, Any] = raw.get("practice", {})
    eval_sec: dict[str, Any] = raw.get("evaluation", {})

    def _req(d: dict, key: str, section: str) -> Any:
        """Return value from *d[key]*, raising KeyError if absent or null."""
        val = d.get(key)
        if val is None:
            raise KeyError(f"Required key '{key}' missing or null in config section [{section}]")
        return val

    practice = PracticeArguments(
        epochs=_req(prac_sec, "epochs", "practice"),
        queries_per_update=_req(prac_sec, "queries_per_update", "practice"),
        grpo_n=_req(prac_sec, "grpo_n", "practice"),
        rollout_concurrency=_req(prac_sec, "rollout_concurrency", "practice"),
        rollout_temperature=_req(prac_sec, "rollout_temperature", "practice"),
        rollout_data_truncate=prac_sec.get("rollout_data_truncate"),
        task_timeout=_req(prac_sec, "task_timeout", "practice"),
        max_retries=_req(prac_sec, "max_retries", "practice"),
        shuffle_data=_req(prac_sec, "shuffle_data", "practice"),
        max_steps_per_rollout=prac_sec.get("max_steps_per_rollout"),
        mcp_result_max_chars=prac_sec.get("mcp_result_max_chars"),
        restart_step=prac_sec.get("restart_step"),
        checkpoint_every=prac_sec.get("checkpoint_every"),
        agent_objective=prac_sec.get("agent_objective"),
        learning_objective=prac_sec.get("learning_objective"),
        given_ground_truth=_req(prac_sec, "given_ground_truth", "practice"),
        num_experiences_per_query=_req(prac_sec, "num_experiences_per_query", "practice"),
        distillation_concurrency=_req(prac_sec, "distillation_concurrency", "practice"),
        plan_library_path=prac_sec.get("plan_library_path"),
        plan_enumeration=bool(prac_sec.get("plan_enumeration", True)),
        do_eval=_req(prac_sec, "do_eval", "practice"),
        eval_strategy=_req(prac_sec, "eval_strategy", "practice"),
        eval_steps=_req(prac_sec, "eval_steps", "practice"),
        eval_data_truncate=prac_sec.get("eval_data_truncate"),
    )

    practice_path_raw = data_sec.get("practice_path") or exp_sec.get("practice_path")
    eval_path_raw = data_sec.get("eval_path")
    data: DataConfig | None = None
    if practice_path_raw:
        data = DataConfig(
            practice_path=Path(practice_path_raw),
            eval_path=Path(eval_path_raw) if eval_path_raw else None,
        )

    evaluation = EvalConfig(
        exp_id=_req(exp_sec, "exp_id", "experiment"),
        concurrency=_req(eval_sec, "concurrency", "evaluation"),
        pass_k=_req(eval_sec, "pass_k", "evaluation"),
        verify_module=eval_sec.get("verify_module"),
        verify_func_name=eval_sec.get("verify_func_name"),
        reward_w_sr=_req(eval_sec, "reward_w_sr", "evaluation"),
        reward_w_row=_req(eval_sec, "reward_w_row", "evaluation"),
        reward_w_item=_req(eval_sec, "reward_w_item", "evaluation"),
        reward_w_cost=_req(eval_sec, "reward_w_cost", "evaluation"),
        reward_w_token=_req(eval_sec, "reward_w_token", "evaluation"),
        reward_w_steps=_req(eval_sec, "reward_w_steps", "evaluation"),
    )

    return PracticeConfig(
        exp_id=_req(exp_sec, "exp_id", "experiment"),
        practice=practice,
        data=data,
        evaluation=evaluation,
        log_dir=exp_sec.get("log_dir") or None,
        # db_files_dir: read from DB_FILES_DIR env var (set in .env).
        # Can be overridden at runtime via --db_files_dir CLI flag.
        db_files_dir=os.environ.get("DB_FILES_DIR") or None,
        system_prompt_path=exp_sec.get("system_prompt_path") or None,
    )

