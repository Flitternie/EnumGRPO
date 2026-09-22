#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
EVAL_ROOT="${ENUMGRPO_EVAL_ROOT:-${REPO_ROOT}/exp/eval/enumgrpo_swan40_n5_v1}"

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${EVAL_ROOT}" \
  --agents enumgrpo \
  --k 3
