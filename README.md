# EnumGRPO

This repository implements **EnumGRPO**, the optimizer from paper [*Cost-Aware Optimization for Agentic Query Execution*](https://arxiv.org/pdf/2606.03152). The project studies database agents that interleave SQL execution with LLM-backed semantic operators, where planning choices affect both answer quality and LLM cost.

EnumGRPO improves an agent through in-context reinforcement learning: during a learning stage it enumerates diverse workflow strategies, executes multiple rollouts per query, scores quality and cost, and distills the contrastive feedback into reusable planning heuristics.

## Method Overview

The optimizer searches over five workflow-planning axes inspired by classical query optimization:

- **Execution paradigm**: data-driven LLM calls vs. code-driven rule synthesis.
- **Operator type**: scalar row-wise operators vs. aggregate/batch operators.
- **Operator placement**: run semantic operators before or after SQL aggregation/filtering.
- **Selectivity scope**: full relation vs. targeted candidate set.
- **Projection width**: narrow key columns vs. wider row context.

Each learning batch runs `grpo_n` rollouts per query, seeded with different axis assignments. Rollouts are scored with execution accuracy, tuple-level F1, cell-level F1, and a group-relative cost penalty over LLM-op tokens and tool steps. A four-stage distillation pipeline updates the experience pool used by the agent at evaluation time.

## Repository Layout

- `agent/`: DB agent runtime and prompts.
- `baseline/`: agentic Text2SQL and BlendSQL baselines.
- `learning/`: EnumGRPO learning loop, reward scoring, and rollout management.
- `swan/`: SWAN JSONL data and evaluation helpers.
- `scalability/`: database scaling and scalability sweep scripts.
- `tools/`: MCP database tools such as SQL execution, relation inspection, and LLM operators.
- `run_multi_eval.sh`: repeated main evaluation across baselines and learned agent.
- `run_heldout.sh`: held-out database learning and evaluation workflow.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate agent
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

Copy `example.env` to `.env` and fill in model credentials and local paths:

```bash
cp example.env .env
```

Required variables for the main experiments:

- `AGENT_MODEL`, `AGENT_API_KEY`, `AGENT_BASE_URL`: planning agent model.
- `LLMOP_MODEL`, `LLMOP_API_KEY`, `LLMOP_BASE_URL`: LLM operator and BlendSQL ingredient model.
- `DB_FILES_DIR`: directory containing SWAN `.duckdb` files.
- `QUERY_TIMEOUT_S`, `QUERY_CONCURRENCY`: run timeout and parallelism controls.
- `LEARNING_LLM_*`: optional separate model for experience distillation; falls back to `AGENT_*`.

## Data

The main SWAN files are:

- `swan/swan.jsonl`: full source dataset.
- `swan/learning.jsonl`: practice data for EnumGRPO.
- `swan/evaluation.jsonl`: evaluation data.
- `swan/database.zip`: bundled SWAN DuckDB database artifact.

The paper uses 40 SWAN questions for learning and 80 for evaluation across four databases: California Schools, European Football, Formula One, and Superhero.

Unzip `swan/database.zip` locally and set `DB_FILES_DIR` in `.env` to the extracted database directory. 

Regenerate the split files from the source dataset with:

```bash
python swan/preprocess.py
```

## Learning

Run the default EnumGRPO learning loop:

```bash
python -m learning.cli --config learning/config.yaml
```

The default config writes results under `exp/learning/<exp_id>/<timestamp>/`. The main learned prompt artifacts are written in that run directory, with latest experience files under `exp/learning/<exp_id>/experiences/`.

Useful overrides:

```bash
python -m learning.cli \
  --config learning/config.yaml \
  --exp_id enumgrpo \
  --practice_path swan/learning.jsonl \
  --eval_path swan/evaluation.jsonl
```

## Main Evaluation

Run repeated evaluation of a learned agent prompt:

```bash
bash run_multi_eval.sh -k 3 \
  -q swan/evaluation.jsonl \
  -e exp/learning/enumgrpo/experiences/latest.json
```

This script evaluates the learned DB agent prompt against the SWAN evaluation set and aggregates repeated runs under `exp/multi_eval_<timestamp>/`. Use `--skip-runs` to re-aggregate existing `eval_summary.json` files.

Run individual baseline agents when needed:

```bash
python run_swan_agentic_text2sql.py --query_file swan/evaluation.jsonl --out_dir exp/text2sql_eval
python run_swan_agentic_blendsql.py --query_file swan/evaluation.jsonl --out_dir exp/blendsql_eval
python run_swan_main.py --query_file swan/evaluation.jsonl --out_dir exp/db_agent_eval
```

Score any run directory with:

```bash
python eval_swan.py --run_dir exp/db_agent_eval --query_file swan/evaluation.jsonl --json
```

## Held-Out DB Evaluation

`run_heldout.sh` supports train-on-three, evaluate-on-one workflows for the four SWAN databases:

```bash
bash run_heldout.sh -k 3
```

The script contains commented sections for generating held-out splits and running per-fold learning jobs. Uncomment the split-generation and fold-learning blocks when producing new held-out learned prompts, then run the evaluation section to score each held-out database.

## Scalability Experiments

Scale the SWAN databases and evaluate agents across scale factors:

```bash
bash scalability/run_scalability_sweep.sh \
  -q swan/evaluation.jsonl \
  -e exp/learning/enumgrpo/experiences/latest.json \
  -k 1
```

Common options:

- `-m text2sql,blendsql,db_agent,db_agent_lx`: choose methods.
- `-s 0.25,0.5,1.0,2.0,4.0`: override scale factors.
- `--db formula_1`: restrict to one database.
- `--skip-scale`: reuse existing scaled DBs.

## Outputs

Most generated artifacts live under `exp/`, including run logs, prompt files, per-query CSVs, `results.jsonl`, `failures.jsonl`, and `eval_summary.json`. The aggregate scripts write summary files in their output directories.
