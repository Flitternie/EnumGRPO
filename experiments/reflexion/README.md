# Reflexion-style baseline

This setup retains EnumGRPO's structured plan enumeration, rollout budget,
reward, and pool-update procedure, but distills each trajectory independently
without within-query grouped comparison. The learning implementation lives in
`learning/reflexion/`.

Train with `sbatch experiments/reflexion/train.sbatch` and evaluate the frozen
artifact pool with `sbatch experiments/reflexion/eval.sbatch`.
