# Cross-workload transfer

These experiments evaluate a frozen SWAN-learned experience pool on workloads
that are not used for learning:

- `sembench/`: all ten queries in SemBench's text-only Movie scenario, plus
  the fixed oracle programs used for the LOTUS, Palimpzest, and BlendSQL
  baselines.
- `spider/`: the Spider 1.0 extra-hard pure-SQL workload.
- `bird/`: the BIRD challenging pure-SQL workload.

Each subdirectory contains its preparation, launch, evaluation, and
aggregation scripts. Downloaded datasets and generated results remain under
`datasets/` and `exp/` and are not committed.
