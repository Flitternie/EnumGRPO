## Training-Free GRPO for the DB Agent

This package implements a **training-free GRPO-style practice loop** around the `agent` (DB agent) runtime.

### Folder Structure

- `config.py` — Dataclass configs loaded from `config.yaml`:
  - `PracticeArguments`: GRPO hyperparameters (`epochs`, `queries_per_update`, `grpo_n`, `rollout_concurrency`, `rollout_temperature`, `distillation_concurrency`, etc.).
  - `DataConfig`: JSONL paths for practice / eval data.
  - `EvalConfig`: evaluation settings — verifier, reward weights (`reward_w_sr`, `reward_w_row`, `reward_w_item`, `reward_w_cost`). **No defaults — all weights must be provided.**
  - `PracticeConfig`: top-level config (`db_files_dir`, `log_dir`, `system_prompt_path`).
- `config.yaml` — **All training hyperparameters live here.** Passed via `--config` at runtime.
- `data_manager.py` — JSONL-backed data loader:
  - Loads base samples (`id` or `dataset_id` field); auto-assigns IDs if omitted.
  - Duplicates each question `grpo_n` times for group rollouts.
  - Slices epoch / batch segments.
- `rollout_manager.py` — Runs the **DB agent** concurrently per question:
  - Calls `run_agent_once(db_path, question, experiences_text)` via `asyncio.to_thread` (true parallelism).
  - Instructs the agent to write its answer to a CSV file; eval reads this CSV.
  - Collects per-action execution profiles (`exec_profile`) — tool name, latency, row counts, token usage, errors.
  - Feeds outputs to the verifier to compute rewards.
  - Records per-batch statistics (`mean_reward`) via `TaskRecorder`.
- `verify.py` — **Multi-objective reward function**:
  - Reads the agent's output CSV (falls back to free-text parsing).
  - Scores with `sr` (success rate), `row_f1`, `item_f1` from `swan/evaluation/utils`.
  - Applies a **linear** LLM-operator token cost penalty (no clamp).
  - **Performance is prioritised**: cost acts as a tiebreaker only when SR/F1 scores are equal within a group.
  - Weights **must** be supplied via config — no defaults; throws an error if missing.
- `experience_updater.py` — Core **training-free GRPO logic** (4-stage LLM pipeline):
  1. **Single-rollout summary** — summarises each trajectory, filtered to groups where `0 < avg_reward < 1` when `given_ground_truth=True`. Includes per-action `exec_profile` for rich signal.
  2. **Group advantage** — compares attempts per question; advantage normalised by group mean and std.
  3. **Group update** — maps each candidate to ADD / UPDATE / DELETE / NONE on the current pool.
  4. **Batch consolidation** — merges all operations into a deduplicated experience set.
  - Uses a **database systems thinking** perspective: prompts focus on data volume, operator cost, logical ordering, and correctness/token trade-offs — without hardcoding specific strategies.
- `cache.py` — `ExperienceCache`: JSON file-based cache under `learning/.cache/<exp_id>/`. Stores both experiences and raw rollouts per step. Supports resuming interrupted runs.
- `enumgrpo.py` — High-level orchestrator (`TrainingFreeGRPO`):
  - Runs the full practice loop with `tqdm` progress bars (epoch → batch → rollouts).
  - `queries_per_update` controls how many queries are rolled out before one distillation update.
  - Automatically clamps `queries_per_update` to the actual dataset size.
  - Uses `math.ceil` so no queries are dropped (partial last batches are processed).
  - Caches rollouts to disk; skips agent re-execution on restart.
- `logging_setup.py` — Two-phase logging:
  - **Phase 1** (`suppress_console_logging`): redirects `sys.stdout`/`sys.stderr` to an in-memory buffer early. Terminal shows only `tqdm` progress bars.
  - **Phase 2** (`setup_run_logging`): flushes buffer to `<log_dir>/<exp_id>/<timestamp>/console.log`, adds JSON and plain-text file handlers.
- `utils.py` — `TaskRecorder`, CLI argument parsing, and `tqdm` progress-bar helpers.
- `cli.py` — CLI entry point: `python -m learning.cli`.

---

### 1. Dataset Format (JSONL)

`DataConfig.practice_path` and `DataConfig.eval_path` must point to **JSONL** files. Each line is a JSON object with at least:

```json
{"id": "q1", "question": "Describe total revenue per month.", "answer": [["Jan", 100], ["Feb", 120]]}
```

- **Required fields**:
  - `id` (or `dataset_id`): string identifier (auto-assigned if omitted).
  - `question`: natural-language task for the DB agent.
- **Optional fields**:
  - `answer`: ground-truth rows (JSON array-of-arrays, string, or `null`). Used by the verifier.
  - `db`: bare database name (e.g. `california_schools`). Resolved as `<DB_FILES_DIR>/<db>.duckdb`. Required for multi-DB datasets.
  - `db_path`: absolute path to a `.duckdb` file. Use when a sample needs a specific file that isn't in `DB_FILES_DIR`.
- Extra keys are preserved in `meta`.

---

### 2. Configuration

All hyperparameters live in `learning/config.yaml`. Pass it with `--config`:

```bash
python -m learning.cli --config learning/config.yaml
```

Key YAML sections and fields:

```yaml
experiment:
  exp_id: my_run
  system_prompt_path: agent/codebase/prompts/agentic_db_plain.md
  log_dir: exp/learning

data:
  practice_path: swan/sample_5.jsonl

practice:
  epochs: 1
  queries_per_update: 5    # queries rolled out before one distillation update
  grpo_n: 5                # rollouts per query (group size)
  rollout_concurrency: 16  # max simultaneous agent runs
  rollout_temperature: 0.7 # LLM temperature for diversity
  distillation_concurrency: 8  # max concurrent LLM calls during distillation
  task_timeout: 3600
  max_retries: 3
  given_ground_truth: true
  num_experiences_per_query: 2

evaluation:
  reward_w_sr: 0.5
  reward_w_row: 0.3
  reward_w_item: 0.2
  reward_w_cost: 0.05
```

Individual fields can be overridden on the command line:

```bash
python -m learning.cli --config learning/config.yaml --grpo_n 3 --log_dir /tmp/logs
```

---

### 3. Environment Variables

Set these in the project-root `.env` file (loaded automatically at startup):

**Agent LLM** (used by `agent`):
- `AGENT_MODEL`
- `AGENT_API_KEY` (not needed for Bedrock)
- `AGENT_BASE_URL` (optional)

**Learning LLM** for experience distillation:
- `LEARNING_LLM_MODEL` (falls back to `AGENT_MODEL`)
- `LEARNING_LLM_API_KEY` (falls back to `AGENT_API_KEY`)
- `LEARNING_LLM_BASE_URL` (falls back to `AGENT_BASE_URL`)

**LLM operators** used inside DB tools (`llm_map`, `llm_reduce`):
- `LLMOP_MODEL`
- `LLMOP_API_KEY`
- `LLMOP_CONCURRENCY` (default: 8)

**Agent runtime**:
- `DB_FILES_DIR` — directory containing `.duckdb` files. When a sample has a `db` field (e.g. `"california_schools"`), the file is resolved as `<DB_FILES_DIR>/california_schools.duckdb`. This is the preferred way to specify databases for multi-DB datasets.

All LLM calls go through `litellm`. Use the `bedrock/` model prefix for AWS Bedrock; set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` as usual.

---

### 4. Multi-Objective Reward

The verifier (`learning.verify.verify_func`) scores each rollout on **performance** and **cost**:

```
reward = w_sr   * sr
       + w_row  * row_f1
       + w_item * item_f1
       - w_cost * (llmop_tokens / 1000)
```

Where:
- `sr` — binary table match (1.0 = exact, 0.0 = wrong)
- `row_f1` — row-level F1 between predicted and ground-truth rows
- `item_f1` — cell-level F1 across all cells
- `llmop_tokens` — tokens consumed by `llm_map`/`llm_reduce` in this rollout

**Design decisions:**
- **Performance-first**: `w_cost` should be small relative to `w_sr + w_row + w_item`. Cost only differentiates rollouts with similar performance scores within the same query group.
- **No clamp**: the cost penalty is linear and unbounded. Excessively token-heavy rollouts receive more negative reward.
- **No defaults**: `EvalConfig` has no fallback weights. Missing weights raise an error at startup.

**Answer parsing** — the agent is instructed to write its answer to a CSV file. The verifier reads that CSV first, then falls back to free-text parsing:
1. CSV output file (preferred)
2. JSON array-of-arrays
3. Markdown / ASCII table
4. CSV inside a code fence / plain CSV
5. Single scalar value

**Advantage computation** is group-relative (per-query, not across queries):
```
advantage_i = (reward_i - mean(group)) / (std(group) + ε)
```

Only groups with `0 < avg_reward < 1` are used for distillation when `given_ground_truth=True`.

---

### 5. Per-Action Execution Profile

Each rollout collects a compact `exec_profile` — one entry per tool call:

```json
[
  {"tool": "open_session",      "duration_s": 0.12, "in_rows": null,  "out_rows": null,  "in_tokens": null, "out_tokens": null, "error": null},
  {"tool": "run_sql",           "duration_s": 0.43, "in_rows": null,  "out_rows": 1240,  "in_tokens": null, "out_tokens": null, "error": null},
  {"tool": "llm_map",           "duration_s": 3.21, "in_rows": 1240,  "out_rows": 1240,  "in_tokens": 8500, "out_tokens": 1240, "error": null},
  {"tool": "close_session",     "duration_s": 0.05, "in_rows": null,  "out_rows": null,  "in_tokens": null, "out_tokens": null, "error": null}
]
```

This profile is passed to the experience distillation pipeline as additional signal for identifying inefficiencies (e.g. large `llm_map` inputs that could be pre-filtered, slow SQL that should be materialised first).

---

### 6. Concurrency Model

```
rollout_concurrency   — max simultaneous agent runs (each in its own thread via asyncio.to_thread)
distillation_concurrency — max concurrent LLM calls during experience distillation
```

With `queries_per_update=5`, `grpo_n=5`, `rollout_concurrency=16`:
- All 25 rollouts (5 queries × 5) are submitted to `asyncio.as_completed` at once.
- A `Semaphore(16)` allows up to 16 to execute simultaneously.
- Each executing rollout runs in its own thread (via `asyncio.to_thread`), so up to 16 agent processes run in parallel.
- Each rollout has its own isolated MCP server subprocess — no shared state between rollouts.
- As each rollout finishes, the next one starts immediately.

**Thread safety**: each `DbAgentRuntime` instance pre-builds its own `ToolDefinition` objects at construction time (with its own MCP client baked in) and injects them directly into `agent._tools` before the conversation starts. This bypasses the global tool registry entirely, making concurrent rollouts safe without any thread-local storage.

---

### 7. Running Training-Free GRPO

**Recommended (YAML config):**

```bash
cd /path/to/db_revise
conda run -n agent python -m learning.cli --config learning/config.yaml
```

`DB_FILES_DIR` and LLM credentials are loaded from `.env` automatically.

**Override individual fields:**

```bash
python -m learning.cli --config learning/config.yaml \
  --exp_id ablation_1 \
  --grpo_n 3 \
  --rollout_concurrency 8
```

**Terminal output** shows only `tqdm` progress bars:
```
Epoch   0/ 1  ████████████████████  100%  [00:02]
  Batch   1/ 1  ████████████████████  100%  [01:45]  step=0
    Rollouts  25/25  ████████████████████  100%  [01:43]  reward=0.512  done=25
```

All logs (agent output, LLM calls, warnings) go to:
```
<log_dir>/<exp_id>/<timestamp>/
  console.log        — everything printed/logged during the run
  learning.log       — structured JSON log
  learning_plain.log — human-readable log
  config_snapshot.yaml — resolved config for reproducibility
  rollouts/          — per-rollout agent run directories
```

---

### 8. Experience Cache

After each batch step, both rollouts and experiences are persisted to:

```
learning/.cache/<exp_id>/
  step_<N>_rollouts.json
  step_<N>_experiences.json
```

On a restart with the same `--exp_id`, the loop loads cached data for completed steps, skipping agent execution and LLM distillation. Delete the cache directory to force a full re-run.

---

### 9. Integrating Experiences into the DB Agent

After the loop completes, the orchestrator writes a versioned system-prompt file:

```
agent/codebase/prompts/agentic_db_<exp_id>.md
```

This is the base `agentic_db.md` with a `## Learned Experiences` section appended:

```markdown
## Learned Experiences (Training-Free GRPO)

[G0]. Schema Inspection: Always use list_relations → describe_relation before querying an unfamiliar table.
[G1]. Cost Control: Materialise filtered rows to a temp table before running llm_map or llm_reduce.
```

The CLI prints the path and the exact flag:

```text
=== Training-free GRPO Experiences ===
[G0] Schema Inspection: Always use list_relations → describe_relation first.
[G1] Cost Control: Materialise filtered rows before llm_map or llm_reduce.

Experienced prompt written to:
  agent/codebase/prompts/agentic_db_my_run.md

To use it with the DB agent CLI:
  python -m codebase db --system_prompt_path agent/codebase/prompts/agentic_db_my_run.md ...
```

**Using a simplified base prompt** — for training-free GRPO to learn from scratch, use the plain prompt that omits handcrafted rules:

```yaml
# learning/config.yaml
experiment:
  system_prompt_path: agent/codebase/prompts/agentic_db_plain.md
```

Each `--exp_id` produces its own file; the base `agentic_db.md` is never modified.
