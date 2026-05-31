#!/usr/bin/env python3
"""DB agent CLI entry point (MCP-backed).

Usage:
    python -m codebase db --db_path /path/to/db.duckdb --message "your question/query" [OPTIONS]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Tools to hide from the agent. Add or remove tool names here to control
# which MCP tools are available during a run.
BLOCKED_TOOLS: list[str] = [
    "column_mapping",
    "row_transform",
    "edit_duckdb",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase",
        description="MCP-backed DuckDB database agent.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- db ---
    db_p = sub.add_parser("db", help="Run MCP-backed DuckDB database agent")
    db_p.add_argument("--db_path", required=True, help="Path to .duckdb file (absolute or relative to repo root)")
    db_p.add_argument("--message", required=True, help="Your natural-language task or SQL question")
    db_p.add_argument(
        "--output_path",
        default=None,
        help="Full path (including filename) to save the final CSV via run_sql(output_path=...).",
    )
    db_p.add_argument(
        "--run_dir",
        default=None,
        help="Override the agent run directory (where logs/artifacts are written). Absolute or relative to repo root.",
    )
    db_p.add_argument("--read_only", default="true", choices=["true", "false"], help="Open DuckDB session read-only (default true)")
    db_p.add_argument("--model", default=None, help="Agent LLM model name (env: AGENT_MODEL)")
    db_p.add_argument(
        "--system_prompt",
        default=None,
        help="Path to a markdown file to use as the system prompt (overrides the default agentic_db.md).",
    )
    db_p.add_argument(
        "--auto",
        action="store_true",
        help="Batch mode: auto-approve risky actions (no confirmation prompts) and exit after one run.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "db":
        from codebase.config import PROJECT_ROOT, RuntimeConfig, get_model_name, get_api_key, get_base_url
        from codebase.runtime import DbAgentRuntime

        # Workspace root for MCP config is the repo root (parent of agent/).
        repo_root = Path(PROJECT_ROOT).resolve().parent

        model = get_model_name(args.model)
        runtime_cfg = RuntimeConfig(model_name=model, api_key=get_api_key(), base_url=get_base_url())

        run_dir_arg = str(args.run_dir).strip() if isinstance(args.run_dir, str) and args.run_dir.strip() else ""
        if run_dir_arg:
            p = Path(run_dir_arg)
            run_dir = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
        else:
            # Include microseconds + pid so concurrent runs don't collide.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"{ts}_db_agent_{os.getpid()}"
            run_dir = Path(PROJECT_ROOT).resolve() / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        rt = DbAgentRuntime(
            workspace_dir=repo_root,
            run_dir=run_dir,
            runtime_cfg=runtime_cfg,
            auto_mode=bool(args.auto),
            system_prompt_path=args.system_prompt or None,
            blocked_tools=BLOCKED_TOOLS or None,
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
                prompt_parts.append("When saving final results, write the CSV EXACTLY to output_path using run_sql(output_path=...).")
            prompt_parts.append("")
            prompt_parts.append("Task:")
            prompt_parts.append(msg)
            prompt = "\n".join(prompt_parts).strip()

            # Pre-open MCP session with the user-provided db_path to avoid LLM mistakes.
            rt.set_db_context(db_path=db_path, read_only=read_only)

            result = rt.run(prompt)
            print(result)

            # Cursor-like experience (default): keep the conversation alive so the user can
            # answer follow-up questions (e.g. provide output_path for CSV export).
            #
            # In --auto mode (batch), exit after one run.
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
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
