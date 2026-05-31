from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence


@dataclass(frozen=True)
class ToolRuntime:
    """
    Runtime context passed from the viewer to tool handlers.
    """

    required_dbs: Sequence[str]
    db_files_dir: str
    db_edit_dir: str
    http_exception_type: Optional[type] = None
    # Optional: when running via the HTTP API with session support enabled.
    # Kept as Any to avoid coupling tools/ to FastAPI or DuckDB types.
    session_manager: Any = None
    session_id: Optional[str] = None
    # Optional callback for reporting progress during long-running operations.
    # Signature: (progress: float, total: Optional[float], message: Optional[str]) -> None
    progress_callback: Optional[Callable[[float, Optional[float], Optional[str]], None]] = None


@dataclass(frozen=True)
class ToolPlugin:
    """
    A plugin defines:
    - a tool schema (for the model)
    - a handler factory (binds ToolRuntime into a callable)
    """

    tool: Dict[str, Any]
    handler_factory: Callable[[ToolRuntime], Callable[[Dict[str, Any]], Any]]

