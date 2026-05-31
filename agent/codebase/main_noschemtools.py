#!/usr/bin/env python3
"""DB agent variant that removes dedicated schema-inspection tools and instead
allows the agent to use `run_sql` with `information_schema` queries for schema
inspection.

Usage:
    python -m codebase.main_noschemtools --db_path /path/to/db.duckdb --message "..."

This is identical to `main.py` except:
- `list_relations`, `describe_relation`, and `preview_relation` are blocked.
- The default system prompt is `agentic_db_noschemtools.md`.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Tell the MCP server subprocess to operate in no-schema-tools mode.
# Must be set before DbAgentRuntime is constructed so the subprocess inherits it.
os.environ.setdefault("MCP_NOSCHEMTOOLS_AGENT", "1")

_HERE = Path(__file__).resolve().parent

BLOCKED_TOOLS: list[str] = [
    "column_mapping",
    "row_transform",
    "edit_duckdb",
    # Schema-inspection tools removed in this variant:
    "list_relations",
    "describe_relation",
    "preview_relation",
    # Never used in practice:
    "explain_sql",
    "profile_query",
]

_DEFAULT_PROMPT = str(_HERE / "prompts" / "agentic_db_noschemtools.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase.main_noschemtools",
        description="MCP-backed DuckDB agent (no dedicated schema-inspection tools).",
    )
    sub = parser.add_subparsers(dest="command")

    db_p = sub.add_parser("db", help="Run the agent")
    db_p.add_argument("--db_path", required=True)
    db_p.add_argument("--message", required=True)
    db_p.add_argument("--output_path", default=None)
    db_p.add_argument("--run_dir", default=None)
    db_p.add_argument("--read_only", default="true", choices=["true", "false"])
    db_p.add_argument("--model", default=None)
    db_p.add_argument(
        "--system_prompt",
        default=None,
        help="Override the default noschemtools prompt.",
    )
    db_p.add_argument("--auto", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "db":
        from codebase.config import PROJECT_ROOT, RuntimeConfig, get_model_name, get_api_key, get_base_url, MAX_ITERATION_PER_RUN
        from codebase.runtime import DbAgentRuntime

        repo_root = Path(PROJECT_ROOT).resolve().parent

        model = get_model_name(args.model)
        runtime_cfg = RuntimeConfig(
            model_name=model,
            api_key=get_api_key(),
            base_url=get_base_url(),
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
        )

        run_dir_arg = str(args.run_dir).strip() if isinstance(args.run_dir, str) and args.run_dir.strip() else ""
        if run_dir_arg:
            p = Path(run_dir_arg)
            run_dir = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"{ts}_db_agent_noschemtools_{os.getpid()}"
            run_dir = Path(PROJECT_ROOT).resolve() / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        system_prompt = args.system_prompt or _DEFAULT_PROMPT

        rt = DbAgentRuntime(
            workspace_dir=repo_root,
            run_dir=run_dir,
            runtime_cfg=runtime_cfg,
            auto_mode=bool(args.auto),
            system_prompt_path=system_prompt,
            blocked_tools=BLOCKED_TOOLS,
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
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
