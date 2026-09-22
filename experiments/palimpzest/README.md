# Palimpzest baseline experiment

This launcher executes the committed oracle-assisted Palimpzest programs three
times and evaluates each run against the SWAN reference answers. The adapter
implementation and fixed programs are documented in `baseline/palimpzest/`.

After checking out the pinned Palimpzest revision described there, run:

```bash
PALIMPZEST_SYSTEM_DIR=/path/to/palimpzest \
  sbatch experiments/palimpzest/run.sbatch
```

Set `PALIMPZEST_OUTPUT_ROOT`, `PYTHON`, or `PALIMPZEST_SYSTEM_DIR` to override
their defaults. Existing output directories are never overwritten.
