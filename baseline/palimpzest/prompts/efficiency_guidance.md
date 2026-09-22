FIXED-PLAN EFFICIENCY (do not sacrifice correctness):
- Push every relational filter, equi-join, grouping, and deduplication that is valid without missing world knowledge before any sem_* operator.
- Before a semantic operator, project to only the semantic input fields plus keys/fields required downstream; do not send irrelevant wide records to the operator LLM.
- Deduplicate identical semantic inputs before sem_map/sem_filter when they can be joined back without changing result multiplicity.
- Use equi-join whenever keys exist. Use sem_join only when no structural join path exists, and reduce both sides to the smallest correct candidate sets first because semantic joins can require O(n*m) model work.
- Prefer sem_agg when one faithful aggregate semantic call can replace many independent record calls; otherwise generate only the missing values needed by the downstream relational computation.
- Never sample, truncate, or add limit merely to save cost when doing so could change the exact answer.
