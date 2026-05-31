#!/usr/bin/env python3
"""BlendSQL agent CLI entry point: exposes run_blendsql only.

This is a thin wrapper around main.py that:
  1. Passes MCP_BLENDSQL_AGENT=1 to the MCP server subprocess so it exposes
     run_blendsql alongside session management (run_sql is not exposed).
  2. Uses the BlendSQL-specific system prompt.

Usage:
    python baseline/agentic_blendsql.py db --db_path /path/to/db.duckdb --message "your query" [OPTIONS]

All other flags (--read_only, --output_path, --model, --auto, etc.) are identical to
the full agent.  Pass --read_only false if the task requires DDL/DML writes.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

# baseline/ sits next to agent/; put agent/ on sys.path so codebase.* is importable.
_AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

# Build the MCP env JSON with BlendSQL agent mode + optional ingredient model config.
_mcp_env: dict[str, str] = {
    "MCP_BLENDSQL_AGENT": "1",
    "LLMOP_MODEL": (os.getenv("LLMOP_MODEL") or "").strip(),
}

os.environ.setdefault("DB_MCP_COMMAND", sys.executable)
os.environ.setdefault("DB_MCP_ARGS_JSON", '["-u", "mcp_server.py"]')
os.environ.setdefault("DB_MCP_ENV_JSON", json.dumps(_mcp_env))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def main() -> None:
    import argparse

    from codebase.config import PROJECT_ROOT, RuntimeConfig, get_model_name, get_api_key, get_base_url
    from codebase.runtime import DbAgentRuntime

    from datetime import datetime

    _BLENDSQL_PROMPT = str(
        (_AGENT_DIR / "codebase" / "prompts" / "agentic_blendsql.md").resolve()
    )

    repo_root = Path(PROJECT_ROOT).resolve().parent

    # Re-use main.py's parser definition to keep CLI flags identical.
    from codebase.main import build_parser

    parser = build_parser()
    args = parser.parse_args()

    if args.command != "db":
        parser.print_help()
        sys.exit(1)

    model = get_model_name(args.model)
    runtime_cfg = RuntimeConfig(model_name=model, api_key=get_api_key(), base_url=get_base_url())

    run_dir_arg = str(args.run_dir).strip() if isinstance(args.run_dir, str) and args.run_dir.strip() else ""
    if run_dir_arg:
        p = Path(run_dir_arg)
        run_dir = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{ts}_blendsql_agent_{os.getpid()}"
        run_dir = Path(PROJECT_ROOT).resolve() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rt = DbAgentRuntime(
        workspace_dir=repo_root,
        run_dir=run_dir,
        runtime_cfg=runtime_cfg,
        auto_mode=bool(args.auto),
        system_prompt_path=_BLENDSQL_PROMPT,
    )
    try:
        db_path = str(args.db_path)
        read_only = str(args.read_only).lower() != "false"
        out_path = str(args.output_path).strip() if isinstance(args.output_path, str) and args.output_path.strip() else ""
        msg = str(args.message)

        prompt_parts = [
            "You are running in CLI mode.",
            f"Database path (db_path): {db_path}",
            f"read_only: {read_only}",
            f"auto: {bool(args.auto)}",
        ]
        if out_path:
            prompt_parts.append(f"Output path for final CSV (output_path): {out_path}")
            prompt_parts.append("When saving final results, write the CSV EXACTLY to output_path using run_blendsql(output_path=...).")
        prompt_parts.append("")
        prompt_parts.append("Task:")
        prompt_parts.append(msg)
        prompt = "\n".join(prompt_parts).strip()

        rt.set_db_context(db_path=db_path, read_only=read_only)

        result = rt.run(prompt)
        print(result)

        if not args.auto and bool(getattr(sys.stdin, "isatty", lambda: False)()):
            while True:
                try:
                    user_msg = input("\nYou (enter to exit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not user_msg or user_msg.lower() in {"exit", "quit", ":q"}:
                    break
                result = rt.run(user_msg)
                print(result)
    finally:
        rt.close()


if __name__ == "__main__":
    main()
