# SemBench text-only transfer experiment

This directory adapts the complete **Movie** scenario from SemBench to the
EnumGRPO agent interface. Movie is SemBench's official table-and-text scenario:
all ten queries are retained, so the subset is scenario-defined rather than
selected from individual queries after observing results.

This is an *agentic adaptation* of SemBench. The original benchmark supplies
system-specific semantic programs to most evaluated engines; here the agent
receives the official natural-language query and constructs the workflow at
runtime. Consequently, results from this directory should not be compared
directly with published SemBench system numbers without noting the interface
difference.

## Oracle baselines

`oracle_programs/` contains the exact fixed programs used for LOTUS,
Palimpzest, and BlendSQL. The LOTUS and Palimpzest files are copied verbatim
from the pinned SemBench revision; the ten BlendSQL files are the query-specific
programs used in our evaluation. See `oracle_programs/README.md` for provenance.
These are called oracle baselines because their workflows are supplied rather
than planned from natural language. Their programs use only the input fields
needed by the queries and do not reference held-out labels or answers.

## Data and visibility policy

The preparation script creates two separate views of the official `sf_2000`
data:

- `sembench_movie.duckdb` is visible to the agent. It omits
  `Reviews.scoreSentiment`, `Reviews.originalScore`, and
  `Reviews.reviewState`, plus `Movies.audienceScore` and
  `Movies.tomatoMeter`. These fields contain ground-truth labels or direct
  answer proxies for the benchmark tasks.
- `ground_truth/Q*.csv` is generated from the untouched official CSV files and
  official gold SQL. It is used only by the evaluator.

The additional removal of movie score columns is necessary for the
natural-language-agent interface: the official Q10 program scores review text,
whereas a free planner could otherwise return `audienceScore` directly.

## Prepare

Download the pinned upstream revision:

```bash
bash experiments/cross_workload/sembench/download_sembench.sh
```

Create the DuckDB database, agent JSONL, ground truth, and audit report:

```bash
python \
  experiments/cross_workload/sembench/prepare_movie.py \
  --sembench-root datasets/sembench/source \
  --output-root datasets/sembench/prepared/movie_sf2000
```

The preparation command refuses to overwrite existing outputs unless
`--force` is supplied.

## Run

The launcher supports three conditions: SQL-only Agentic Text2SQL, Agentic
Query Execution without experiences, and Agentic Query Execution with the
frozen SWAN-learned experiences. Set `SEMBENCH_RUNS=1` for a pilot or `3` for
the final repeated experiment.

```bash
SEMBENCH_RUNS=1 sbatch experiments/cross_workload/sembench/run_movie_conditions.sbatch
```

`SEMBENCH_CONDITIONS` may be set to `text2sql` or `agents` to run only that
part of the comparison; its default is `all`.

Run a fixed LOTUS or Palimpzest program with its baseline environment:

```bash
SEMBENCH_SYSTEM=lotus \
SEMBENCH_QUERIES=1,2,3,4,5,6,7,8,9,10 \
SEMBENCH_OUTPUT_DIR=exp/eval/sembench_movie/baselines_same_llm/lotus/run_1 \
sbatch experiments/cross_workload/sembench/run_oracle_programs.sbatch
```

Run the ten vendored BlendSQL programs:

```bash
sbatch experiments/cross_workload/sembench/run_blendsql.sbatch
```

Every condition is evaluated with SemBench's query-specific metric family:
retrieval F1 for Q1/Q2/Q5--Q7, relative error for Q3/Q4/Q8, and Spearman and
Kendall rank correlation for Q9/Q10. The evaluator also reports wall time and
the repository's planner/LLM-operator usage accounting.

SemBench's Q6 gold output contains both pair orientations near its file head.
Because the official evaluator normalizes unordered pairs after truncating to
ten rows, even a gold-output replay scores 0.947 rather than 1.0 F1 on Q6. The
adapter intentionally preserves this upstream behavior.

To evaluate a completed run manually:

```bash
python \
  experiments/cross_workload/sembench/evaluate_movie.py \
  --run-dir exp/eval/sembench_movie/agent_no_pool/run_1 \
  --prepared-root datasets/sembench/prepared/movie_sf2000 \
  --output exp/eval/sembench_movie/agent_no_pool/run_1/eval_summary.json
```

## Interpretation

The scenario has only ten queries but 2,000 review rows and requires semantic
processing in every query. It is therefore useful as a complementary semantic
transfer diagnostic, not as evidence of a larger query count. Q10 semantically
scores all 2,000 reviews and is expected to dominate cost.
