# BIRD pure-SQL transfer experiment

This experiment evaluates the agent without additional learning on the 231
challenging questions in the BIRD development split. The preparation step
converts the source SQLite databases to DuckDB and audits each translated gold
query before producing the agent-facing JSONL.

Prepare and audit the workload:

```bash
sbatch experiments/cross_workload/bird/prepare_challenging.sbatch
sbatch experiments/cross_workload/bird/audit_challenging.sbatch
```

Run the first repetition, then repetitions 2--5 and aggregate them:

```bash
sbatch experiments/cross_workload/bird/run_challenging_run1.sbatch
sbatch experiments/cross_workload/bird/run_challenging_runs2_5.sbatch
sbatch experiments/cross_workload/bird/aggregate_challenging_k5.sbatch
```

The jobs compare an agent without experiences against one using the frozen
SWAN-learned pool. Gold SQL is used only to construct and audit reference
answers; it is not exposed to the agent.
