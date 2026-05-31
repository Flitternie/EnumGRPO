# CLI Agent for Data Analytics

This folder contains a small CLI “database agent” that talks to DuckDB **only** via the repo’s MCP tool server (`mcp_server.py`). It’s meant for interactive analytics (and optional data repair) using the MCP tool surface (`run_sql`, `list_relations`, `llm_map`, etc.).

## Quickstart

### Prerequisites

- **Python**: a local environment with the agent’s runtime deps available.
- **An LLM API key**:
  - Set `AGENT_API_KEY` in your shell (used by the main agent).
  - If you use LLM operators like `llm_map` / `llm_reduce`, also set `LLMOP_API_KEY` (used by the MCP tools).

### Install (minimal)

Install the dependencies in your environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install duckdb python-dotenv pydantic mcp openhands
```

## Run the agent

Run from inside `agent/` so `codebase` is importable as a module.

```bash
cd agent
python -m codebase db \
  --db_path "<path/to/file.duckdb>" \
  --message "List the tables, then show 5 rows from the most relevant one."
```

### Common options

- **Export a final CSV**

```bash
cd agent
python -m codebase db \
  --db_path "<path/to/file.duckdb>" \
  --message "Compute total revenue by month." \
  --output_path "/tmp/revenue_by_month.csv"
```

- **Batch mode (no prompts, one-shot)**: auto-approves risky actions and exits after one run.

```bash
cd agent
python -m codebase db \
  --db_path "<path/to/file.duckdb>" \
  --message "Run the requested query and save it." \
  --output_path "/tmp/out.csv" \
  --auto
```

- **Read-write sessions**: by default the agent opens DuckDB **read-only**. If you set `--read_only false`, the MCP server enforces that the DB path must be under `SANDBOX_DIR` (defaults to `./sandbox`-style edited area).

```bash
cd agent
python -m codebase db \
  --db_path "<path/under/db_edit_dir/file.duckdb>" \
  --read_only false \
  --message "Apply the edits and verify the result."
```

- **Model / endpoint overrides**
  - Agent model is **required** (no default): set `--model` or `AGENT_MODEL`.
  - CLI (agent LLM): `--model "openai/gpt-4o"`
  - Env (agent LLM): `AGENT_MODEL`
  - Env (LLM operators `llm_map` / `llm_reduce`): `LLMOP_MODEL` (on the MCP server side)
  - Optional base URL: `AGENT_BASE_URL`

## MCP server configuration (how tools are found)

The agent runtime (`codebase/runtime.py`) resolves MCP server settings in this order:

- **Explicit env vars**:
  - `DB_MCP_COMMAND`
  - `DB_MCP_ARGS_JSON` (JSON list)
  - `DB_MCP_ENV_JSON` (JSON object)
- Otherwise it reads the repo’s **`.cursor/mcp.json`** (from the repo root; `agent/` is a subfolder).
  - If `DB_MCP_SERVER_NAME` is set, it prefers that server name.
- Otherwise it falls back to: `python -u mcp_server.py` (from the repo root).

## Outputs and logs

- **Agent run artifacts**: written under `agent/runs/<timestamp>_db_agent/`
  - Includes conversation state and LLM completion logs.
- **MCP session logs**: `mcp_server.py` writes per-session JSONL logs under `logs/` at the repo root.

## Troubleshooting

- **`Environment variable AGENT_API_KEY ... is not set`**
  - Export `AGENT_API_KEY`.
- **`LLMOP API key not configured (set LLMOP_API_KEY)`**
  - Export `LLMOP_API_KEY`.
- **MCP client can’t start / no tools**
  - Confirm `.cursor/mcp.json` points to a working Python and that `mcp_server.py` runs in that env.
- **`duckdb python package is not installed`**
  - Install `duckdb` into the environment used to run `mcp_server.py`.

