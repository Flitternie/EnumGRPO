# Palimpzest SWAN baseline

This is a strict two-stage, declarative Palimpzest baseline. Program generation happens once, and every measured repetition executes the exact same saved program.

The three-run paper setup is available at `experiments/palimpzest/run.sbatch`.

## Code structure

- `generate_queries.py`: stage 1 Sonnet translator. It builds the planner prompt from the live DuckDB schema and sample rows; `--oracle` optionally adds gold SQL and BlendSQL; the resulting fixed `build_pipeline(db)` program is saved to JSONL.
- `run.py`: stage 2 execution harness. It reads fixed programs, executes them K times, and never imports planner code or prompt files.
- `common.py`: shared DuckDB loading, PZ configuration, program construction, serialization, hashing, and usage-accounting helpers.
- `pz_integration.py`: runtime adapters for model routing, infrastructure retries, token accounting, usage journaling, and offline model metadata.
- `prompts/`: planner, oracle, and repair prompt templates used only by stage 1.
- `programs/swan_oracle_efficiency.jsonl`: the fixed 80-query program set used
  for the reported oracle-assisted result.
- `.env.example`: optional Palimpzest-only settings that have no repository-level common-variable equivalent.

The third-party Palimpzest source is intentionally not vendored. Check out the
audited upstream revision and either place it at `baseline/palimpzest/system`
or set `PALIMPZEST_SYSTEM_DIR`:

```bash
git clone https://github.com/mitdbg/palimpzest.git /path/to/palimpzest
git -C /path/to/palimpzest checkout 807ed301c4d2457ef304647e6095184556bd1e83
export PALIMPZEST_SYSTEM_DIR=/path/to/palimpzest
python -m pip install -e "$PALIMPZEST_SYSTEM_DIR"
```

All EnumGRPO integration code remains in this directory and leaves the
upstream checkout unmodified.

## Stage 1: generate fixed programs

Fair generation:

```bash
python baseline/palimpzest/generate_queries.py --efficiency-guidance
```

Oracle-construction generation, with and without efficiency guidance:

```bash
python baseline/palimpzest/generate_queries.py --oracle --efficiency-guidance
python baseline/palimpzest/generate_queries.py --oracle --no-efficiency-guidance
```

`--oracle` exists only in stage 1. The efficiency flag is required so the variant is never ambiguous. Default filenames include both dimensions, for example `generated_queries_oracle_efficiency.jsonl` and `generated_queries_oracle_no_efficiency.jsonl`; use `--out_file` to override the path.

Each record contains the question metadata, final program, program SHA-256, generation error, generation attempts, and diagnostic generation wall time. Stage 1 time, tokens, and cost are excluded from execution metrics.

## Stage 2: execute fixed programs

```bash
python baseline/palimpzest/run.py \
  --program_file baseline/palimpzest/programs/swan_oracle_efficiency.jsonl \
  --out_dir exp/baseline/palimpzest/oracle_efficiency \
  --num_runs 3
```

This creates `run_r1` through `run_rK`. Stage 2 does not load planner prompts, repair code, or regenerate programs. Each run records the source JSONL SHA-256 and every selected program hash in `program_manifest.json`.

Each run directory contains `<question_id>.csv`, `<question_id>.usage.json`, `pz_manifest.jsonl`, agentic-compatible `results.jsonl` and `failures.jsonl`, and `program_manifest.json`.

Each task writes its CSV and usage immediately, then atomically checkpoints `pz_manifest.jsonl`, `results.jsonl`, and `failures.jsonl`. The task log uses the same `ok`, `returncode`, `elapsed_s`, `stdout`, `stderr`, and `job` structure as the agentic SWAN runner. Add `--resume` to keep completed results whose program hashes match and rerun only missing, infrastructure-failed, fatal/output-failed, corrupt, or mismatched tasks. Resume is applied independently to every `run_rK`.

After a completed benchmark run, use `--resume --retry_infra_only` to replace only model/API infrastructure failures in place. This preserves measured timeouts, OOMs, code/type failures, and child exits while rebuilding the complete per-run manifests and task logs.

## Measurement and failure accounting

Per-question wall time covers DuckDB loading, PZ optimization, and fixed-program execution. Planner generation is excluded. Operator cost is recomputed with the shared Haiku price sheet: $1/M input, $0.10/M cache read, $1.25/M cache write, and $5/M output.

Normal code errors, timeouts, and OOMs remain measured failures and retain completed operator usage. Model/API infrastructure retries are excluded. PZ's native execution cache is scoped to the execution-strategy instance; each query runs in a new isolated child process, so the cache is automatically reset at every query boundary. The usage journal recovers completed calls after timeout, OOM, or hard kill. Full tables are used when `--row_cap 0`.

## Configuration

Common data, model, endpoint, credential, concurrency, timeout, and accounting settings come from the repository-level `.env` through `DB_FILES_DIR`, `AGENT_*`, `LLMOP_*`, `AWS_*`, `QUERY_CONCURRENCY`, and `E1_*`. A local `baseline/palimpzest/.env` is optional and should contain only Palimpzest-specific settings from `.env.example`.

`LLMOP_CONCURRENCY` directly sets Palimpzest's per-query execution worker pool (`QueryProcessorConfig.max_workers`). `QUERY_CONCURRENCY` separately controls how many queries the stage-2 harness executes at once.

`E1_MAX_TRANSLATION_ATTEMPTS` defaults to 3, `E1_MAX_QUERY_GENERATION_RETRIES` defaults to 3 with a fixed `E1_QUERY_GENERATION_RETRY_DELAY_S` of 5 seconds, `E1_ROW_SAMPLE_SIZE` defaults to 20, `E1_ROW_SAMPLE_MAX_CHARS` defaults to 6000 per table, `E1_MAX_INFRA_ATTEMPTS` defaults to 3, common `QUERY_TIMEOUT_S` defaults to 1800, and `PZ_TABLE_ROW_CAP` defaults to 0.

## Offline smoke test

```bash
python baseline/palimpzest/generate_queries.py \
  --mock --efficiency-guidance --limit 1 \
  --out_file /tmp/pz_programs.jsonl

python baseline/palimpzest/run.py \
  --mock --program_file /tmp/pz_programs.jsonl \
  --out_dir /tmp/pz_runs --num_runs 2 --concurrency 1
```
