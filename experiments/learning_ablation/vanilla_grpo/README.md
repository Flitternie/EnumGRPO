# Vanilla GRPO baseline

This setup disables EnumGRPO's structured plan enumeration and uses
temperature sampling while retaining the same grouped comparison, reward,
learning queries, and rollout budget.

Train with
`sbatch experiments/learning_ablation/vanilla_grpo/train.sbatch` and evaluate
the frozen artifact pool with
`sbatch experiments/learning_ablation/vanilla_grpo/eval.sbatch`.
