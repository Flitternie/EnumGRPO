# Evaluation runners

These reusable command-line runners execute the agent and agentic baselines
over the repository's JSONL workload format:

- `agent.py`: query-execution agent, with or without an experience pool.
- `agentic_text2sql.py`: Agentic Text2SQL baseline.
- `agentic_blendsql.py`: Agentic BlendSQL baseline.

Experiment-specific launchers live under `experiments/` and call these
runners with their own datasets, pools, and output directories.
