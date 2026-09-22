# LOTUS baseline experiment

This launcher executes the committed oracle-assisted LOTUS programs three
times and evaluates each run against the SWAN reference answers. The adapter
implementation and fixed programs are documented in `baseline/lotus/`.

After checking out the pinned LOTUS revision described there, run:

```bash
LOTUS_SYSTEM_DIR=/path/to/lotus sbatch experiments/lotus/run.sbatch
```

Set `LOTUS_OUTPUT_ROOT`, `PYTHON`, or `LOTUS_SYSTEM_DIR` to override their
defaults. Existing output directories are never overwritten.
