"""
Training-free GRPO utilities for the DB agent.

This package provides a self-contained implementation of the core ideas from
`utu/practice` adapted to the `db_revise` agent stack:

- Config models for practice and data loading
- A rollout manager that drives the DB agent as the underlying policy
- An experience updater that distills group-relative advantages into
  reusable "experiences" (instructions)
- A high-level `TrainingFreeGRPO` orchestrator that ties everything together

The goal is to keep dependencies on the existing `utu` codebase minimal while
preserving the overall algorithmic structure.
"""

import sys
from pathlib import Path

# Ensure the project root (db_revise/) and agent/ are on sys.path before any
# submodule imports. This must happen here rather than in cli.py because
# __init__.py is executed first when running `python -m learning.cli`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # db_revise/
_AGENT_ROOT   = _PROJECT_ROOT / "agent"                  # db_revise/agent/
for _p in (str(_PROJECT_ROOT), str(_AGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Phase 1: silence console logging before heavy imports (openhands, litellm,
# etc.) can add their own StreamHandlers.
from .logging_setup import suppress_console_logging
suppress_console_logging()

from .config import PracticeConfig, PracticeArguments, DataConfig
from .enumgrpo import TrainingFreeGRPO
from .utils import TaskRecorder

__all__ = [
    "PracticeConfig",
    "PracticeArguments",
    "DataConfig",
    "TrainingFreeGRPO",
    "TaskRecorder",
]

