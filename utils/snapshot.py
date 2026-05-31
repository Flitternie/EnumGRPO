"""Utilities for writing a config snapshot at the start of each run."""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_CREDENTIAL_KEYS = frozenset({
    "key", "token", "password",
})

_ENV_KEYS = [
    "AGENT_MODEL", "AGENT_BASE_URL",
    "AGENT_MAX_ITERATIONS",
    "LLMOP_MODEL", "LLMOP_BASE_URL",
    "DB_FILES_DIR",
    "QUERY_TIMEOUT_S", "QUERY_CONCURRENCY",
    "LLMOP_TIMEOUT_S", "LLMOP_CONCURRENCY",
    "MCP_RESULT_MAX_CHARS",
    "BLENDSQL_ASYNC_LIMIT", "BLENDSQL_ROW_LIMIT",
]


def _repo_root_from_script(script: str) -> Path:
    return Path(script).resolve().parent


def _read_agent_limits(repo_root: Path) -> Dict[str, Any]:
    limits: Dict[str, Any] = {}
    try:
        # Import is relative to repo root; add to path temporarily if needed.
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from agent.codebase.config import MAX_ITERATION_PER_RUN  # type: ignore
        limits["max_iteration_per_run"] = MAX_ITERATION_PER_RUN
    except Exception:
        pass
    try:
        runtime_src = (repo_root / "agent" / "codebase" / "runtime.py").read_text()
        m = re.search(r"self\.tool_text_content_limit\s*=\s*(\d+)", runtime_src)
        if m:
            limits["tool_text_content_limit"] = int(m.group(1))
    except Exception:
        pass
    return limits


def _git_commit(cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except Exception:
        return None


def save_config_snapshot(
    out_dir: Path,
    *,
    script: str,
    args_dict: Dict[str, Any],
    extra_sections: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``config_snapshot.json`` to *out_dir* with all non-credential run config."""
    repo_root = _repo_root_from_script(script)

    env_snapshot = {k: os.environ.get(k) for k in _ENV_KEYS if os.environ.get(k) is not None}

    safe_args = {
        k: v for k, v in args_dict.items()
        if not any(cred in k.lower() for cred in _CREDENTIAL_KEYS)
    }

    snapshot: Dict[str, Any] = {
        "script": script,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(repo_root),
        "args": safe_args,
        "env": env_snapshot,
        "agent_limits": _read_agent_limits(repo_root),
    }
    if extra_sections:
        snapshot.update(extra_sections)

    try:
        (out_dir / "config_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Warning: could not write config_snapshot.json: {exc}", file=sys.stderr)
