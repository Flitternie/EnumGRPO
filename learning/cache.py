"""Simple JSON-based cache for training-free GRPO experiences.

This mirrors the behavior of `utu.utils.ExperienceCache` but stores data in
local JSON files under `learning/.cache/`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent / ".cache"


@dataclass
class ExperienceCacheRecord:
    experiment_name: str
    step: int
    epoch: int | None
    batch: int | None
    experiences: dict[str, Any]


class ExperienceCache:
    """File-based cache for experiences per (experiment, step)."""

    @staticmethod
    def _step_path(experiment_name: str, step: int) -> Path:
        safe_exp = experiment_name.replace("/", "_")
        exp_dir = BASE_DIR / safe_exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir / f"step_{step}.json"

    @staticmethod
    def _rollouts_path(experiment_name: str, step: int) -> Path:
        safe_exp = experiment_name.replace("/", "_")
        exp_dir = BASE_DIR / safe_exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir / f"step_{step}_rollouts.json"

    @classmethod
    def save_experiences(
        cls,
        experiment_name: str,
        step: int,
        experiences: dict[str, Any],
        *,
        epoch: int | None = None,
        batch: int | None = None,
    ) -> None:
        rec = ExperienceCacheRecord(
            experiment_name=experiment_name,
            step=step,
            epoch=epoch,
            batch=batch,
            experiences=experiences,
        )
        path = cls._step_path(experiment_name, step)
        data = {
            "experiment_name": rec.experiment_name,
            "step": rec.step,
            "epoch": rec.epoch,
            "batch": rec.batch,
            "experiences": rec.experiences,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_experiences(cls, experiment_name: str, step: int) -> dict[str, Any] | None:
        path = cls._step_path(experiment_name, step)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupt experiences cache at %s; will re-run step.", path)
            return None
        exps = raw.get("experiences")
        if not isinstance(exps, dict):
            logger.warning("Invalid experiences cache at %s (unexpected format); will re-run step.", path)
            return None
        return exps

    @classmethod
    def save_rollouts(
        cls,
        experiment_name: str,
        step: int,
        rollouts: list[Any],
    ) -> None:
        """Persist serialised rollout results for a step."""
        from dataclasses import asdict

        path = cls._rollouts_path(experiment_name, step)
        path.write_text(
            json.dumps([asdict(r) for r in rollouts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_rollouts(cls, experiment_name: str, step: int) -> list[Any] | None:
        """Load previously-saved rollout results (as plain dicts).

        Returns ``None`` if no cache file exists or the file is corrupt.
        Returns a list of dicts (not ``RolloutResult`` instances) to keep the
        cache module free of circular imports.
        """
        path = cls._rollouts_path(experiment_name, step)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupt rollouts cache at %s; will re-run step.", path)
            return None
        if not isinstance(raw, list):
            logger.warning("Invalid rollouts cache at %s (expected list); will re-run step.", path)
            return None
        return raw

