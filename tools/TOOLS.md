# Tools interface (concise)

This repo exposes tools via the MCP server (`mcp_server.py`). Most tools require a `session_id` from `open_session`.

## Session management (MCP)

### `open_session`
- **Purpose**: Open a persistent DuckDB connection (enables TEMP objects + multi-step work).
- **Input**: `db_path | db_name(+db_root)`, `db_files_dir`, `db_edit_dir`, `read_only: bool`
- **Output**: `{ ok, session_id, db_path, read_only, db_files_dir, db_edit_dir, required_dbs }`

### `close_session`
- **Purpose**: Close a session/connection.
- **Input**: `session_id`
- **Output**: `{ ok, closed, session_id }`

### `list_sessions`
- **Purpose**: Enumerate open sessions.
- **Input**: none
- **Output**: `{ ok, sessions: [...] }`

### `set_session_roots`
- **Purpose**: Update `db_files_dir` / `db_edit_dir` for an existing session.
- **Input**: `session_id`, `db_files_dir?`, `db_edit_dir?`
- **Output**: `{ ok, session_id, db_files_dir, db_edit_dir }`

## Read-only SQL + inspection

### `run_sql`
- **Purpose**: Execute a **read-only** `SELECT/WITH` and preview results (row-capped).
- **Input**: `sql`, `db_path?`, `limit_rows?`, `output_path?`, `include_header?`
- **Output**: `{ db_files?, columns, rows, row_count, limit_rows, saved? }`

### `explain_sql`
- **Purpose**: `EXPLAIN` / `EXPLAIN ANALYZE` for a read-only query (plan-first safety).
- **Input**: `sql`, `analyze?`
- **Output**: `{ analyze, statement, plan_text?, columns, rows, db_files }`
- **Note**: requires `session_id`.

### `list_relations`
- **Purpose**: Discover candidate tables/views when the schema is ambiguous.
- **Input**: `schema?`, `like?`, `include_system?`, `limit?`, `db_path?`
- **Output**: `run_sql`-style table of `{table_schema, table_name, table_type}`.

### `describe_relation`
- **Purpose**: Inspect a table/view’s columns/types/nullability (+ best-effort PK hint).
- **Input**: `relation` (`table` or `schema.table`), `db_path?`
- **Output**: `{ relation, columns: <run_sql-style rows>, primary_key_hint? }`

### `preview_relation`
- **Purpose**: Preview example rows from a table/view to disambiguate ambiguous column names (and spot obvious corruption).
- **Input**: `relation`, `columns?`, `where?`, `limit_rows?`, `db_path?`
- **Output**: `run_sql`-style rows.

### `materialize_temp`
- **Purpose**: Create a session-scoped **TEMP** table/view from a `SELECT/WITH` for stepwise workflows.
- **Input**: `name`, `kind: table|view`, `sql`, `replace?`, `row_count?`
- **Output**: `{ ok, name, kind, statement, row_count?, schema, db_files }`
- **Note**: requires `session_id`.

### `profile_query`
- **Purpose**: Lightweight data-quality profile for a small query result (corruption/mixed-format signals).
- **Input**: `sql`, `limit_rows?`, `max_cell_chars?`, `db_path?`
- **Output**: `{ ok, query: {limit_rows}, profile: { row_count, columns: [...] } }`
- **Per-column signals**: null rate, distinct/top values (sample), numeric + ISO-datetime parse rates (sample), length stats, examples.

## Write operations (persistent or session-writable)

### `edit_duckdb`
- **Purpose**: Apply ordered DDL/DML steps to an editable DB copy under `db_edit/` (or a writable session DB).
- **Input**: `sql_steps[]`, `target_db_path?`, `dest_db_name?`, `preview_sql?`, `limit_rows?`
- **Output**: `{ ok, edited_db_path, cloned_from?, sql_log, statements_executed, preview? }`

### `row_transform`
- **Purpose**: Deterministic per-row writeback using a key column (client computes updates).
- **Modes**:
  - `action="fetch"`: read `key_column` + selected `columns`
  - `action="apply"`: apply `updates[{key,value}]` into `target_column` (optionally create it)
- **Input**: `table`, `key_column`, `where?`, `limit_rows?`, plus apply-specific fields
- **Output**: fetch returns `run_sql`-style rows; apply returns `edit_duckdb`-style result.

### `column_mapping`
- **Purpose**: Apply a provided value→value mapping to a column (replace or write to new column).
- **Input**: `table`, `source_column`, `target_column?`, `create_column?`, `new_column_type?`, `mapping`, `unmapped_behavior?`, `where?`, edit targets + `preview_sql?`
- **Output**: `edit_duckdb`-style result.

## LLM operators

### `llm_map`
- **Purpose**: LLM-based normalization/mapping and write-back to a table column.
- **Flow**: `action="plan"` → returns `plan_id` + estimates; then `action="run"` with `confirm=true` + `plan_id`.
- **Inputs (common)**: `instruction`, `output_type?`, `options?/options_sql?`, `mode: distinct|per_row`, write target config.
- **Output**: plan returns estimates; run returns write result + `llm_map` stats + `mapping_preview`.

### `llm_reduce`
- **Purpose**: LLM-based reasoning over a context query result (reduce rows → one answer).
- **Flow**: `action="plan"` → returns `plan_id` + estimates; then `action="run"` with `confirm=true` + `plan_id`.
- **Input**: `question`, `context_sql`, `output_type?`, `options?`, `limit_rows?`, `max_cell_chars?`
- **Output**: `{ answer_text, answer, context, prompt_stats, ... }`

