"""Ablation variant of TrainingFreeGRPO: independent rollout distillation.

This module is a minimal wrapper around the standard training pipeline that
swaps out the GRPO experience updater for the independent variant
(:class:`IndependentExperienceUpdater`).  Everything else — rollout execution,
plan enumeration, caching, checkpointing, eval, and the CLI — is reused without
modification.

Usage (same flags as the standard CLI):

    python -m learning.ablation_ungrouped.cli --config learning/config.yaml [overrides...]
"""

from __future__ import annotations

from pathlib import Path

from .experience_updater import IndependentExperienceUpdater
from ..enumgrpo import TrainingFreeGRPO


class TrainingFreeIndependent(TrainingFreeGRPO):
    """Training-free distillation that treats every rollout independently.

    Identical to :class:`TrainingFreeGRPO` except that the
    :class:`IndependentExperienceUpdater` is used in place of the standard
    :class:`ExperienceUpdater`.  No group-relative z-score advantages are
    computed; each rollout is distilled on its own merits using absolute scores.
    """

    def __init__(self, config, *, run_log_dir: Path | None = None) -> None:
        super().__init__(config, run_log_dir=run_log_dir)
        # Replace the GRPO updater installed by the parent __init__.
        self._experience_updater = IndependentExperienceUpdater(config)
