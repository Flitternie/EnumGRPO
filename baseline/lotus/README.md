# LOTUS SWAN baseline

This is a strict two-stage, non-agentic LOTUS baseline. Program generation happens once, and every measured repetition executes the exact same saved program.

The three-run paper setup is available at `experiments/lotus/run.sbatch`.

## Code structure

- `generate_queries.py`: stage 1 Sonnet translator. It builds the planner prompt from the SWAN question, schema, and sample rows; `--oracle` optionally adds gold SQL and BlendSQL; the resulting fixed program is saved to JSONL.
- `run.py`: stage 2 execution harness. It reads fixed programs, executes them K times, and never imports or calls the planner.
- `common.py`: shared SWAN loading, DuckDB schema, serialization, normalization, and hashing helpers.
- `prompts/`: planner, oracle, repair, and semantic-operator prompt templates.
- `programs/swan_oracle_efficiency.jsonl`: the fixed 80-query program set used
  for the reported oracle-assisted result.

The third-party LOTUS source is intentionally not vendored. Check out the
audited upstream revision and either place it at `baseline/lotus/system` or set
`LOTUS_SYSTEM_DIR`:

```bash
git clone https://github.com/lotus-data/lotus.git /path/to/lotus
git -C /path/to/lotus checkout 136ae4f4a344a2f75d89f811e516dfcb0de30e46
export LOTUS_SYSTEM_DIR=/path/to/lotus
python -m pip install -e "$LOTUS_SYSTEM_DIR"
```

All EnumGRPO integration code remains in this directory and leaves the
upstream checkout unmodified.

## Stage 1: generate fixed programs

Fair generation:

```bash
python baseline/lotus/generate_queries.py --efficiency-guidance
```

Oracle-construction generation, with and without efficiency guidance:

```bash
python baseline/lotus/generate_queries.py --oracle --efficiency-guidance
python baseline/lotus/generate_queries.py --oracle --no-efficiency-guidance
```

`--oracle` exists only in stage 1. The efficiency flag is required so the variant is never ambiguous. Default filenames include both dimensions, for example `generated_queries_oracle_efficiency.jsonl` and `generated_queries_oracle_no_efficiency.jsonl`; use `--out_file` to override the path.

Each record contains the question metadata, final program, program SHA-256, generation error, generation attempts, and diagnostic generation wall time. Stage 1 time, tokens, and cost are excluded from execution metrics.

## Stage 2: execute fixed programs

```bash
python baseline/lotus/run.py \
  --program_file baseline/lotus/programs/swan_oracle_efficiency.jsonl \
  --out_dir exp/baseline/lotus/oracle_efficiency \
  --num_runs 3
```

This creates `run_r1` through `run_rK`. Stage 2 does not load planner prompts, repair code, or regenerate programs. Each run records the source JSONL SHA-256 and every selected program hash in `program_manifest.json`.

Each run directory contains `<question_id>.csv`, `<question_id>.usage.json`, aggregate `usage.json`, agentic-compatible `results.jsonl` and `failures.jsonl`, and `program_manifest.json`.

Each task writes its CSV and usage immediately, then atomically checkpoints aggregate `usage.json`, `results.jsonl`, and `failures.jsonl`. The task log uses the same `ok`, `returncode`, `elapsed_s`, `stdout`, `stderr`, and `job` structure as the agentic SWAN runner. Add `--resume` to keep completed results whose program hashes match and rerun only missing, infrastructure-failed, fatal, corrupt, or mismatched tasks. Resume is applied independently to every `run_rK`.

## Measurement and failure accounting

Per-question wall time covers DuckDB loading and fixed-program execution. Planner generation is excluded. Operator cost is recomputed with the shared Haiku price sheet: $1/M input, $0.10/M cache read, $1.25/M cache write, and $5/M output.

Normal code errors and timeouts remain measured failures and retain completed operator usage. Model/API infrastructure retries are excluded. LOTUS model/operator caching is enabled during each query and reset before and after every query, preventing cross-query and cross-run reuse. Physical token usage and cache-hit counts are reported. Tables are loaded in full, and the timeout comes from the common `QUERY_TIMEOUT_S` setting (1800 seconds when unset).

## Configuration

All LOTUS baseline settings are covered by repository-level common variables, so no baseline-local `.env` is needed. Stage 1 uses `PLANNER_*` with `AGENT_*` fallbacks; stage 2 uses `LLMOP_*`; both stages use `DB_FILES_DIR`, `E1_*`, and the standard `AWS_*` variables where applicable.

`E1_MAX_TRANSLATION_ATTEMPTS` defaults to 3, `E1_MAX_QUERY_GENERATION_RETRIES` defaults to 3 with a fixed `E1_QUERY_GENERATION_RETRY_DELAY_S` of 5 seconds, `E1_ROW_SAMPLE_SIZE` defaults to 20, `E1_ROW_SAMPLE_MAX_CHARS` defaults to 6000 per table, `E1_MAX_INFRA_ATTEMPTS` defaults to 3, and `E1_TOKEN_CONVENTION` defaults to `physical`.

## Offline smoke test

```bash
python baseline/lotus/generate_queries.py \
  --dry_run --efficiency-guidance --limit 1 \
  --out_file /tmp/lotus_programs.jsonl

python baseline/lotus/run.py \
  --dry_run --program_file /tmp/lotus_programs.jsonl \
  --out_dir /tmp/lotus_runs --num_runs 2
```
