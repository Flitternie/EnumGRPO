# Primary EnumGRPO experiment

This directory contains the training configuration and launchers for the
primary SWAN experiment. Shared execution and evaluation infrastructure remains
at the repository root.

- Train: `sbatch experiments/primary/train.sbatch`
- Evaluate the released pool: `sbatch experiments/primary/eval.sbatch`
- Re-aggregate completed runs: `bash experiments/primary/summarize.sh`

The evaluation launcher reads the frozen pool from
`artifacts/experiences/swan_enumgrpo.json`.
