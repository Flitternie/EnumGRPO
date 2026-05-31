"""Data loading utilities for training-free GRPO.

Unlike the original `utu` implementation which relies on a SQLModel-backed
database, this module works directly with JSONL files. Each line in the file
must be a JSON object with at least:

    {
        "id": str,
        "question": str,
        "answer": str | null
    }

Additional fields are preserved and passed through to downstream components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import PracticeConfig


PracticeStage = Literal["init", "rollout", "judged"]


@dataclass
class PracticeSample:
    """In-memory representation of a single practice / eval sample."""

    id: str
    question: str
    answer: str | None
    meta: dict[str, Any]

    # Fields that are filled during the rollout / judging process.
    stage: PracticeStage = "init"
    trajectories: list[dict[str, Any]] | None = None
    reward: float | None = None
    reasoning: str | None = None

    # Plan-enumeration fields: injected by RolloutManager.load_epoch_data when
    # plan_enumeration is enabled.  plan_hint is the assembled strategy hint
    # string; plan_axes is the raw axis-value assignment dict for distillation
    # labeling.
    plan_hint: str | None = None
    plan_axes: dict[str, Any] | None = None


class JsonlDataManager:
    """JSONL-backed analogue of `TrainingFreeGRPODataManager`.

    It handles:
    - Loading the base dataset from a JSONL file
    - Duplicating samples according to pass_k / grpo_n
    - Slicing per-epoch and per-batch segments.
    """

    def __init__(self, config: PracticeConfig) -> None:
        if config.data is None:
            raise ValueError("PracticeConfig.data must be set.")
        self.config = config
        self._base_samples: list[PracticeSample] = self._load_base_samples(config.data.practice_path)

    @classmethod
    def from_path(cls, path: Path, config: PracticeConfig) -> "JsonlDataManager":
        """Create a manager backed by an arbitrary JSONL path.

        Useful for loading the eval dataset independently of the practice path
        stored in config.data.
        """
        instance = object.__new__(cls)
        instance.config = config
        instance._base_samples = instance._load_base_samples(path)
        return instance

    def _load_base_samples(self, path: Path) -> list[PracticeSample]:
        if not path.exists():
            raise FileNotFoundError(f"Practice dataset JSONL not found: {path}")
        samples: list[PracticeSample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q_id = str(obj.get("id") or obj.get("question_id") or obj.get("dataset_id") or len(samples))
                question = str(obj.get("question") or obj.get("query") or "")
                if not question:
                    raise KeyError(f"No 'question' or 'query' field found in record: {list(obj.keys())}")
                answer = obj.get("answer")
                meta = {k: v for k, v in obj.items() if k not in {"id", "question_id", "dataset_id", "question", "query", "answer"}}
                samples.append(
                    PracticeSample(
                        id=q_id,
                        question=question,
                        answer=answer,
                        meta=meta,
                    )
                )
        if not samples:
            raise ValueError(f"Practice dataset at {path} is empty.")
        return samples

    def load_base_samples(self, *, truncate: int | None = None) -> list[PracticeSample]:
        """Return the base samples (1 per question, no grpo_n duplication).

        Used for evaluation passes where a single rollout per question is enough.
        """
        base = self._base_samples
        if truncate is not None:
            base = base[:truncate]
        return list(base)

    def load_epoch_data(self, epoch: int, *, truncate: int | None = None, shuffle: bool = True) -> list[PracticeSample]:
        """Create a duplicated list of samples for the given epoch.

        Each base question is duplicated `grpo_n` times so that we can form
        groups of attempts per query.
        """
        import random

        base = self._base_samples
        if truncate is not None:
            base = base[:truncate]

        if shuffle:
            # Use a per-epoch seed so batches are identical on restart (e.g. when
            # resuming with restart_step).  Epoch 0 → seed 0, epoch 1 → seed 1, …
            base = list(base)
            random.Random(epoch).shuffle(base)

        out: list[PracticeSample] = []
        for s in base:
            for k in range(self.config.practice.grpo_n):
                duplicated = PracticeSample(
                    id=f"{s.id}::attempt_{k}",
                    question=s.question,
                    answer=s.answer,
                    meta=s.meta,
                )
                out.append(duplicated)
        return out

    @staticmethod
    def get_batch(
        epoch_samples: list[PracticeSample],
        *,
        batch_idx: int,
        grpo_n: int,
        queries_per_update: int,
    ) -> list[PracticeSample]:
        """Return the subset of samples belonging to a (epoch, batch) pair."""
        group_size = grpo_n * queries_per_update
        start = batch_idx * group_size
        end = start + group_size
        return epoch_samples[start:end]

