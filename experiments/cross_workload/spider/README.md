# Spider 1.0 pure-SQL robustness experiment

This experiment tests whether the SWAN-learned experience pool transfers to a
new, fully relational workload without causing unnecessary LLM-operator calls.

The recommended protocol starts from the 340 hard/extra-hard questions in
Spider 1.0 dev.  This keeps the experiment focused on structurally complex SQL
instead of letting easy single-table questions dominate.  Gold SQL is used
only to construct reference answers, difficulty strata, and a pre-model
DuckDB-compatibility audit; it is never included in an agent prompt.

## Data preparation

```bash
bash experiments/cross_workload/spider/download_spider.sh

python \
  experiments/cross_workload/spider/prepare_spider.py \
  --archive datasets/spider1/raw/spider_data.zip \
  --upstream-root datasets/spider1/raw/upstream \
  --output-root datasets/spider1/prepared \
  --sample-size 240 \
  --seed 463
```

Spider's databases and gold SQL target SQLite.  Do not assume that successful
parsing or dialect transpilation implies equivalent DuckDB behavior.  After
conversion, audit every candidate by executing both versions and comparing
their result tables with the repository's evaluation semantics:

```bash
python \
  experiments/cross_workload/spider/audit_duckdb_compat.py \
  --source-root datasets/spider1/prepared/source/spider_data \
  --duckdb-dir datasets/spider1/prepared/database \
  --output-report datasets/spider1/prepared/compatibility_raw.json \
  --output-jsonl datasets/spider1/prepared/evaluation_hard_extra_compatible.jsonl
```

`--dialect sqlglot` can be used for a generic transpilation pass, and
`--dialect normalized` additionally applies the two corpus-audited Spider
normalizations (double-quoted literals and a deterministic rewrite of SQLite's
permissive GROUP BY by adding non-aggregate projected expressions).
Compatibility must still depend on equal executed results, not merely on
successful translation.  Exclude only cases that still cannot be matched;
filtering all raw failures would disproportionately remove GROUP BY questions
and bias the benchmark toward easier SQL.

Build the extra-hard DuckDB evaluation file first.  The `preserve` policy keeps
the official Spider semantics, including known annotation mistakes, while
making only the syntax/type changes required by DuckDB.  As in SWAN, boundary
ties are represented by `answer_b`, `answer_c`, etc.:

```bash
python \
  experiments/cross_workload/spider/build_duckdb_eval.py \
  --source-root datasets/spider1/prepared/source/spider_data \
  --duckdb-dir datasets/spider1/prepared/database \
  --hardness extra \
  --semantic-policy preserve \
  --output-jsonl datasets/spider1/prepared/evaluation_extra_duckdb.jsonl \
  --output-report datasets/spider1/prepared/duckdb_extra_report.json
```

The file contains all 166 extra-hard questions, including 26 with multiple
valid answer tables.  No question is excluded and no official semantic error
is corrected.  The original Spider SQL remains in `sql`; its DuckDB-compatible
equivalent is stored separately in `duckdb_gold_sql`.

Two agent-integration details must be addressed before a formal run:

- The converter currently preserves tables, values, and row counts, but not
  primary/foreign-key declarations.  Supply the official `tables.json` schema
  relationships to every compared method (or preserve them as DuckDB metadata).
- Some hard/extra-hard answers exceed the tool's default 200-row preview.  The
  final CSV path must save the complete result (all such answers are below
  1,000 rows); preview truncation must not become evaluation truncation.

Generate the isolated agent-facing JSONL with official PK/FK metadata:

```bash
python \
  experiments/cross_workload/spider/prepare_agent_eval.py \
  --input-jsonl datasets/spider1/prepared/evaluation_extra_duckdb.jsonl \
  --tables-json datasets/spider1/prepared/source/spider_data/tables.json \
  --output-jsonl datasets/spider1/prepared/evaluation_extra_duckdb_agent.jsonl
```

Spider jobs set `RUN_SQL_FULL_CSV_EXPORT=1` and
`RUN_SQL_FULL_CSV_MAX_ROWS=1000`.  This keeps the MCP result shown to the
agent bounded while writing the complete final query result to CSV.  Both
variables are opt-in, so the historical SWAN execution path is unchanged.

The downloaded dataset is CC BY-SA 4.0. The official Spider evaluator code is
Apache-2.0. Downloaded and converted assets are ignored by Git.

## Planned comparisons

1. SQL-only Text2SQL baseline.
2. Agent without an experience pool.
3. Agent with the frozen SWAN experience pool, without Spider retraining.

Report execution accuracy, row/item F1, LLM-operator invocation rate, token
cost, and wall time. Formal timing runs must pass the NFS/runtime preflight and
all compared systems must use the same execution environment.

The agent-facing metadata also instructs every method to save an output CSV
for empty results.  Fifteen extra-hard questions have an empty gold answer;
missing output files remain failures, while a saved header-only CSV is scored
as the intended empty result.  Evaluation summaries record both token-bearing
LLM-op queries and the exact invocation count, so failed/zero-token calls are
not mistaken for pure-SQL execution.
