# Learning ablations

These experiments isolate two components of EnumGRPO's learning procedure:

- `vanilla_grpo/` removes structured plan enumeration while retaining
  grouped comparison.
- `reflexion/` retains structured enumeration but removes grouped comparison
  by updating from each trajectory independently.

Both use the same SWAN learning queries and evaluation protocol as the primary
experiment. Experience-category deletion is kept separately in
`experiments/axis_ablation/` because it studies the learned pool's contents
rather than the learning algorithm.
