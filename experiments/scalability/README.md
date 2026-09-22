# SWAN scalability experiment

This experiment deterministically scales each SWAN database from `0.25x` to
`4x` and evaluates the configured agent variants at every scale. Generated
databases and run outputs are written under `exp/`.

Run the complete sweep from the repository root:

```bash
bash experiments/scalability/run_scalability_sweep.sh \
  -q swan/evaluation.jsonl \
  -e artifacts/experiences/swan_enumgrpo.json \
  -k 1
```

Use `--list` with `scale_databases.py` to inspect a scale plan without writing
databases. See `run_scalability_sweep.sh --help` for method, scale, and timeout
options.
