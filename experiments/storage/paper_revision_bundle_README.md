# Paper Revision Experiment Bundle

This bundle contains the clean experiment setups and selected results used by
the revised manuscript and response letter. Each setup has separate `code/`,
`inputs/`, and `results/` directories. Original workspace files are not moved
or modified.

## Included setups

| Directory | Purpose | Reported runs |
|---|---|---:|
| `lotus_oracle_assisted` | Oracle-assisted LOTUS baseline | 3 |
| `palimpzest_oracle_assisted` | Oracle-assisted Palimpzest baseline | 3 |
| `vanilla_grpo` | EnumGRPO without plan enumeration | 3 evaluation runs |
| `reflexion` | EnumGRPO without grouped comparison | 3 evaluation runs |
| `gpt_backbone_transfer` | GPT backbone transfer, with and without the experience pool | 3 per condition |
| `pure_sql_spider` | Spider 1.0 pure-SQL transfer, with and without the experience pool | 3 per condition |
| `gold_oracle_routing` | Single-run gold-oracle routing diagnostic for the response letter | 1 paired run |

The result directories retain their original run-level logs, per-query JSONL
records, CSV outputs, and aggregate summaries. Generated LOTUS and Palimpzest
programs are stored under each setup's `inputs/` directory because they are
fixed experimental inputs rather than handwritten implementation code.

## External system revisions

The local wrappers were run against these unmodified third-party checkouts:

- LOTUS: commit `136ae4f4a344a2f75d89f811e516dfcb0de30e46`
- Palimpzest: commit `807ed301c4d2457ef304647e6095184556bd1e83`

Their full repositories and Python environments are not duplicated here. The
wrapper code, prompts, generated programs, environment specification, and
recorded outputs needed to audit the reported experiments are included.

## Deliberate exclusions

The bundle excludes smoke tests, failed diagnostics, incomplete runs,
unselected extra runs under `legacy/`, the unused no-pool thinking ablation,
Python caches, virtual environments, downloaded database files, credentials,
and pre-existing archives. Spider preparation manifests and evaluation JSONL
files are included, while the public Spider database files can be regenerated
with the included preparation scripts.

The bundle is scoped to revision-added experiments and diagnostics. It does
not duplicate the original main SWAN, scaling, or cross-database experiments
whose raw run directories are not part of the clean revision result tree.
