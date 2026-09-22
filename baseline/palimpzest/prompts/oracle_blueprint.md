=== REFERENCE BLUEPRINT (ground-truth QUERY STRUCTURE — for constructing the pipeline correctly; it is NOT the answer and contains NO answer rows) ===
The correct result is defined by this gold SQL over the ORIGINAL COMPLETE database. SWAN has DROPPED some columns from the live database, so columns referenced here may NOT exist in the schema above — those must be recovered by a semantic operator.
GOLD SQL:
{sql}

The correct HYBRID decomposition is this gold BlendSQL. Every `llm`/`llm_*` table or join marks EXACTLY the column(s) that were DROPPED and must be RECOVERED by an LLM (route these through a sem_map / sem_join / sem_filter operator keyed on the join column(s) shown). Everything else (plain filters, joins on real key columns, group-by, aggregation) is relational.
GOLD BLENDSQL:
{blendsql}
Faithfully reproduce THIS query's logic in Palimpzest. If the blueprint uses ORDER BY ... LIMIT N (N>1) or another construct PZ cannot express (no sort / no numeric top-N), build the most faithful pipeline the PZ ops allow — never hardcode which rows win, and never hardcode any literal data value from the blueprint. Use it for STRUCTURE only.
