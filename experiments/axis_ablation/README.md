# Experience-axis ablation

This experiment measures the contribution of each category in the frozen
32-entry EnumGRPO experience pool. The supplied manual mapping assigns every
experience to exactly one primary plan-space axis and matches the counts and
polarities reported in the paper.

The experiment uses leave-one-axis-out pools:

- `without_execution_paradigm` (removes 9, retains 23)
- `without_operator_type` (removes 3, retains 29)
- `without_operator_placement` (removes 6, retains 26)
- `without_selectivity_scope` (removes 8, retains 24)
- `without_projection_width` (removes 6, retains 26)

The full-pool condition is not rerun by default; compare against the existing
three-run full EnumGRPO result. All five new conditions use the original SWAN
evaluation set, Claude environment, four-query concurrency, 1,800-second query
timeout, local compute-node database staging, and three independent runs.

Submit with:

```bash
bash experiments/axis_ablation/submit.sh
```

For a one-run validation, set `AXIS_ABLATION_RUNS=1` before submission. To use
a separate result root, set `AXIS_ABLATION_OUTPUT_ROOT` consistently for the
submission command and exported Slurm environment.
