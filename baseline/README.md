# Baselines

The repository includes four baseline integrations:

- `agentic_text2sql.py`: Agentic Text2SQL, using an LLM planner but no explicit
  LLM operator during query execution.
- `agentic_blendsql.py` and `blendsql/`: Agentic BlendSQL.
- `lotus/`: the EnumGRPO SWAN adapter for LOTUS.
- `palimpzest/`: the EnumGRPO SWAN adapter for Palimpzest.

LOTUS and Palimpzest are evaluated with fixed, oracle-assisted programs. Their
complete generated programs and generation manifests are committed under each
adapter's `programs/` directory. The third-party frameworks are not vendored;
their READMEs identify the exact upstream commits and setup procedure.
