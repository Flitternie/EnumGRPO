# Cross-database transfer

This experiment performs four leave-one-database-out folds on SWAN. For each
fold, EnumGRPO learns from the other three databases and is evaluated only on
the held-out database.

Run all training and three-run evaluation folds with:

```bash
bash experiments/cross_db/run.sh
```

Use `--skip-training` to evaluate existing fold-specific pools or
`--skip-evaluation` to produce the pools only.
