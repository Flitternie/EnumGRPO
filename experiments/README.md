# Experiments

This directory contains the reproducible launchers and configurations used by
the paper and revision experiments:

- `primary/`: EnumGRPO learning and evaluation on SWAN, including launchers
  for the main-table baselines.
- `learning_ablation/`: Vanilla GRPO and Reflexion learning ablations.
- `cross_db/`: leave-one-database-out transfer on SWAN.
- `cross_workload/`: transfer to SemBench Movie and pure-SQL Spider/BIRD.
- `axis_ablation/`: experience-axis deletion experiments.
- `scalability/`: SWAN database scaling and evaluation sweep.

Each subdirectory documents its own configuration, launch commands, and output
layout. Frozen experience pools used by these experiments are stored in
`artifacts/experiences/`.

Generated runs and logs belong under `exp/` and are not part of the source
layout. Cluster-specific storage audits and archive-building utilities are also
excluded because they are operational maintenance scripts rather than
experimental methods.
