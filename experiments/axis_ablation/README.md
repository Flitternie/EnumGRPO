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

The full-pool condition is not rerun by default; compare against the primary
three-run EnumGRPO result. The runner uses the original SWAN evaluation set,
four-query concurrency, a 1,800-second query timeout, and compute-node-local
database staging.

Submit with:

```bash
AXIS_ABLATION_ENV_FILE=.env.gpt \
AXIS_ABLATION_OUTPUT_ROOT=exp/eval/experience_axis_ablation_gpt_swan_v1 \
bash experiments/axis_ablation/submit.sh
```

`AXIS_ABLATION_RUN_IDS` controls the Slurm array (default `1-3`), while
`AXIS_ABLATION_CONDITIONS` selects one or more conditions. For example, a
supplemental run can be submitted without a one-off script using
`AXIS_ABLATION_RUN_IDS=4-5`, one condition, and
`AXIS_ABLATION_ALLOW_EXISTING=1`. Each run directory is still protected from
overwriting.

The source pool defaults to
`artifacts/experiences/swan_enumgrpo.json`; override it with
`AXIS_ABLATION_SOURCE_POOL`. Aggregate completed GPT runs with:

```bash
python experiments/axis_ablation/summarize.py \
  --base-dir exp/eval/experience_axis_ablation_gpt_swan_v1 \
  --runs 3 --gpt-pricing
```
