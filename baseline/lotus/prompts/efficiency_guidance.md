FIXED-PLAN EFFICIENCY (do not sacrifice correctness):
  - Push every relational filter, key join, grouping, and deduplication that is valid without missing world knowledge before any sem_* call.
  - Before a semantic call, project to only the semantic input columns plus keys/columns required downstream; do not send irrelevant wide rows to the operator LM.
  - Deduplicate identical semantic inputs before sem_map/sem_filter when they can be joined back without changing result multiplicity.
  - Use pandas key joins whenever keys exist. Use sem_join only when no structural join path exists, and reduce both sides to the smallest correct candidate sets first.
  - Prefer sem_agg when one faithful aggregate semantic call can replace many independent row calls; otherwise generate only the missing values needed by the downstream relational computation.
  - Never sample, truncate, or add LIMIT merely to save cost when doing so could change the exact answer.
