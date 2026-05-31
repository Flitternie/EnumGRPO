"""Logging configuration for a training-free GRPO run.

Two-phase design
----------------
1. ``suppress_console_logging()`` — call as early as possible (before heavy
   imports).  Redirects ``sys.stdout`` and ``sys.stderr`` to an in-memory
   buffer so that *all* terminal chatter (logging, print(), rich Console,
   Python warnings) is captured rather than shown.  Python preserves the
   original terminal as ``sys.__stderr__``, which tqdm uses explicitly.

2. ``setup_run_logging(cfg)`` — call after config is resolved.  Flushes the
   early buffer to ``<run_log_dir>/console.log``, then redirects
   stdout/stderr to that file for the rest of the run.  Also adds structured
   JSON and plain-text handlers to the Python logging system.

Terminal output after phase 1: only tqdm progress bars (which write to
``sys.__stderr__`` directly) and explicit ``tqdm.write()`` calls.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    from .config import PracticeConfig

# ---------------------------------------------------------------------------
# Module-level state for the two-phase redirect
# ---------------------------------------------------------------------------
_early_buffer: io.StringIO | None = None
_log_file: IO[str] | None = None
_run_file_handlers: list[logging.FileHandler] = []  # handlers added by the last setup call


# ---------------------------------------------------------------------------
# Known noisy loggers — set to WARNING so debug chatter doesn't bloat logs
# ---------------------------------------------------------------------------
_NOISY_LOGGERS = [
    "LiteLLM", "LiteLLM Proxy", "LiteLLM Router",   # capital-L names used by litellm
    "litellm", "litellm.utils", "litellm.proxy",
    "httpx", "httpcore",
    "openai", "openai._base_client",
    "boto3", "botocore",
    "urllib3",
    "asyncio",
    "openhands", "openhands.sdk",
    "mcp",
    "rich",
]


def suppress_console_logging() -> None:
    """Phase 1 — redirect all console output to an in-memory buffer.

    Safe to call multiple times; subsequent calls are idempotent.
    ``sys.__stderr__`` always points at the original terminal and is used
    by tqdm so progress bars still appear.
    """
    global _early_buffer
    if _early_buffer is not None:
        return  # already called

    # Redirect stdout/stderr to buffer.
    # sys.__stdout__ / sys.__stderr__ remain untouched (Python internals).
    _early_buffer = io.StringIO()
    sys.stdout = _early_buffer
    sys.stderr = _early_buffer

    # Route Python warnings into logging (captured by file handlers later).
    logging.captureWarnings(True)
    warnings.filterwarnings("default")

    # Fix root logger level so file handlers added later receive all records.
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # Floor noisy third-party loggers.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


class _PlainFormatter(logging.Formatter):
    FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def setup_run_logging(cfg: "PracticeConfig") -> Path | None:
    """Phase 2 — flush early buffer to file and add structured log handlers.

    Returns the run log directory, or ``None`` when ``cfg.log_dir`` is falsy.
    """
    global _early_buffer, _log_file

    if not cfg.log_dir:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = Path(cfg.log_dir) / cfg.exp_id / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Redirect stdout/stderr to console.log ----------------------------
    console_log_path = run_log_dir / "console.log"
    _log_file = console_log_path.open("w", encoding="utf-8", buffering=1)

    if _early_buffer is not None:
        _log_file.write(_early_buffer.getvalue())
        _early_buffer = None

    sys.stdout = _log_file
    sys.stderr = _log_file

    # Uncaught exceptions must appear on the terminal even though sys.stderr
    # is redirected.  This hook writes the traceback to both destinations.
    import traceback as _tb

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.__stderr__, flush=True)
        if _log_file and not _log_file.closed:
            print(msg, file=_log_file, flush=True)

    sys.excepthook = _excepthook

    # --- Python logging file handlers ------------------------------------
    root_logger = logging.getLogger()

    # Remove any StreamHandlers added during heavy imports.
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)

    json_handler = logging.FileHandler(run_log_dir / "learning.log", encoding="utf-8")
    json_handler.setFormatter(_JsonFormatter())
    json_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(json_handler)

    plain_handler = logging.FileHandler(run_log_dir / "learning_plain.log", encoding="utf-8")
    plain_handler.setFormatter(_PlainFormatter())
    plain_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(plain_handler)

    # Track so teardown_run_logging() can close them later.
    _run_file_handlers.clear()
    _run_file_handlers.extend([json_handler, plain_handler])

    # --- Config snapshot --------------------------------------------------
    _write_config_snapshot(cfg, run_log_dir / "config_snapshot.yaml")

    # Use sys.__stderr__ (original terminal) so this line appears on screen.
    print(f"Run logs → {run_log_dir}", file=sys.__stderr__, flush=True)
    logging.getLogger(__name__).info("Run log directory: %s", run_log_dir)
    return run_log_dir


def _write_config_snapshot(cfg: "PracticeConfig", path: Path) -> None:
    try:
        import yaml
        import dataclasses

        def _to_dict(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            if isinstance(obj, Path):
                return str(obj)
            return obj

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_to_dict(cfg), f, allow_unicode=True, sort_keys=False)
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not write config snapshot: %s", exc)


def teardown_run_logging() -> None:
    """Close and remove the FileHandlers added by the most recent setup_run_logging call.

    Call this after each fold/run completes to prevent file-descriptor accumulation
    when multiple folds are run sequentially in the same process (e.g. heldout training).
    Safe to call even if setup_run_logging was never called.
    """
    global _log_file, _run_file_handlers

    root_logger = logging.getLogger()
    for handler in list(_run_file_handlers):
        try:
            root_logger.removeHandler(handler)
            handler.close()
        except Exception:
            pass
    _run_file_handlers.clear()

    # Flush and close the console.log redirect; restore stdout/stderr to the
    # original terminal so the next fold's setup_run_logging starts clean.
    if _log_file is not None:
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
        _log_file = None

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
