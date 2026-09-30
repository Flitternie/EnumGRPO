# Primary EnumGRPO experiment

This directory contains the training configuration and launchers for the
primary SWAN experiment.

- Train: `sbatch experiments/primary/train.sbatch`
- Evaluate the released pool: `sbatch experiments/primary/eval.sbatch`
- Run the evaluation orchestration directly:
  `bash experiments/primary/run_multi_eval.sh`
- Re-aggregate completed runs: `bash experiments/primary/summarize.sh`
- Run oracle-assisted baselines:
  `experiments/primary/baselines/{lotus,palimpzest}/`

The evaluation launcher reads the frozen pool from
`artifacts/experiences/swan_enumgrpo.json`. Reusable Python execution
entrypoints are under `runners/`.
