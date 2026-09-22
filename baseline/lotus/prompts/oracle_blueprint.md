=== REFERENCE BLUEPRINT (ground-truth QUERY STRUCTURE — for constructing the pipeline correctly; it is NOT the answer, and contains NO answer rows) ===
The correct result is defined by this gold SQL over the ORIGINAL COMPLETE database. NOTE: SWAN has DROPPED some columns from the live database, so columns referenced here may NOT exist in the schema above — those must be recovered by a semantic operator.
GOLD SQL:
{sql}

The correct HYBRID decomposition is this gold BlendSQL. Every `llm`/`llm_*` table or join marks EXACTLY the column(s) that were DROPPED and must be RECOVERED by an LLM (route these through a sem_* operator keyed on the join column(s) shown). Everything else (plain filters, joins on real key columns, group-by, aggregation, ORDER BY / top-N) is relational — do it with plain pandas exactly as the SQL specifies.
GOLD BLENDSQL:
{blendsql}
Faithfully reproduce THIS query's logic in LOTUS. Do NOT hardcode any literal data value taken from the blueprint; use it for STRUCTURE only (which columns are relational vs LLM-recovered, and how they combine).
