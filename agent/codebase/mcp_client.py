from __future__ import annotations

import asyncio
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import mcp.types as mcp_types


@dataclass(frozen=True)
class McpServerConfig:
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: Optional[str] = None
    # Timeout (seconds) for individual tool calls and other RPC round-trips.
    call_timeout_sec: Optional[float] = None


class McpStdioClient:
    """
    Long-lived MCP stdio client with a dedicated event loop thread, so synchronous
    callers (OpenHands tool executors) can call MCP tools safely.
    """

    def __init__(self, cfg: McpServerConfig) -> None:
        self._cfg = cfg
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-stdio-client")
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None

        # Serialises concurrent start() calls so the thread is only started once.
        self._start_lock = threading.Lock()

        self._stdio_cm = None
        self._session_cm = None
        self._session: Optional[ClientSession] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._main_task: Optional[asyncio.Task] = None
        self._errlog_file: Optional[IO[str]] = None

    def start(self, *, timeout_sec: float = 20.0) -> None:
        with self._start_lock:
            if self._thread.is_alive():
                return
            # A Python Thread can only be started once.  If the thread was
            # started before and has since exited (e.g. the MCP subprocess
            # crashed or close() was called mid-rollout due to a timeout),
            # we must recreate both the event loop and the thread before
            # trying again, otherwise Thread.start() raises
            # "RuntimeError: threads can only be started once".
            if self._thread.ident is not None:
                # Close the old event loop before replacing it to avoid the
                # "unclosed event loop" ResourceWarning.
                try:
                    if not self._loop.is_closed():
                        self._loop.close()
                except Exception:
                    pass
                self._loop = asyncio.new_event_loop()
                self._ready = threading.Event()
                self._init_error = None
                self._stdio_cm = None
                self._session_cm = None
                self._session = None
                self._stop_evt = None
                self._main_task = None
                self._thread = threading.Thread(
                    target=self._run_loop, daemon=True, name="mcp-stdio-client"
                )
            self._thread.start()
        if not self._ready.wait(timeout=timeout_sec):
            raise TimeoutError("Timed out starting MCP client")
        if self._init_error:
            raise RuntimeError(f"Failed to start MCP client: {self._init_error}") from self._init_error

    def list_tools(self) -> List[mcp_types.Tool]:
        self.start()
        return self._run(self._list_tools_async())

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> mcp_types.CallToolResult:
        self.start()
        args = arguments if isinstance(arguments, dict) else None
        return self._run(self._call_tool_async(name, args), timeout=self._cfg.call_timeout_sec)

    def __enter__(self) -> "McpStdioClient":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self, *, timeout_sec: float = 30.0) -> None:
        if not self._thread.is_alive():
            return
        try:
            self._signal_stop()
            self._run(self._wait_closed_async(), timeout=timeout_sec)
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

    # -------------------------
    # Internals
    # -------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._stop_evt = asyncio.Event()
        self._main_task = self._loop.create_task(self._main())
        self._loop.run_forever()

    async def _main(self) -> None:
        env = dict(os.environ)
        env.update(self._cfg.env or {})
        params = StdioServerParameters(
            command=self._cfg.command,
            args=list(self._cfg.args or []),
            env=env,
            cwd=self._cfg.cwd,
        )
        # Route the MCP server process's stderr to a per-rollout file so that
        # concurrent rollouts don't interleave output on the terminal.
        # Falls back to sys.stderr when RUNNING_DIR is not set (e.g. interactive use).
        running_dir = (self._cfg.env or {}).get("RUNNING_DIR", "").strip()
        errlog: IO[str]
        if running_dir:
            stderr_path = Path(running_dir) / "logs" / "mcp_server" / "stderr.log"
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._errlog_file = stderr_path.open("a", encoding="utf-8", buffering=1)
            errlog = self._errlog_file
        else:
            errlog = sys.stderr
        try:
            self._stdio_cm = stdio_client(params, errlog=errlog)
            read_stream, write_stream = await self._stdio_cm.__aenter__()

            self._session_cm = ClientSession(read_stream, write_stream)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
        except BaseException as e:
            self._init_error = e
            self._ready.set()
            return

        self._ready.set()

        # Keep the context managers alive until close() signals stop.
        assert self._stop_evt is not None
        await self._stop_evt.wait()

        # Best-effort: tear down in reverse order in the SAME TASK that entered.
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
        finally:
            self._session = None
            self._session_cm = None
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
            if self._errlog_file is not None:
                try:
                    self._errlog_file.close()
                except Exception:
                    pass
                self._errlog_file = None

    async def _list_tools_async(self) -> List[mcp_types.Tool]:
        if self._session is None:
            raise RuntimeError("MCP session is not initialized")
        res = await self._session.list_tools()
        return list(res.tools or [])

    async def _call_tool_async(self, name: str, args: Dict[str, Any] | None) -> mcp_types.CallToolResult:
        if self._session is None:
            raise RuntimeError("MCP session is not initialized")
        return await self._session.call_tool(name=name, arguments=args or {})

    def _signal_stop(self) -> None:
        if self._stop_evt is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        except Exception:
            pass

    async def _wait_closed_async(self) -> None:
        if self._main_task is None:
            return
        try:
            await self._main_task
        except Exception:
            # Do not raise close-time errors to callers.
            return

    def _run(self, coro, *, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

