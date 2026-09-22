#!/usr/bin/env bash
# Train EnumGRPO on three SWAN databases and evaluate on the held-out fourth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="${REPO_ROOT}/experiments/primary/config.yaml"
TRAIN_FILE="${REPO_ROOT}/swan/learning.jsonl"
EVAL_FILE="${REPO_ROOT}/swan/evaluation.jsonl"
SPLIT_ROOT="${REPO_ROOT}/exp/learning/cross_db_splits"
OUTPUT_ROOT="${CROSS_DB_OUTPUT_ROOT:-${REPO_ROOT}/exp/eval/cross_db_swan}"
K="${CROSS_DB_RUNS:-3}"
SKIP_TRAINING=0
SKIP_EVALUATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --runs) K="$2"; shift 2 ;;
    --skip-training) SKIP_TRAINING=1; shift ;;
    --skip-evaluation) SKIP_EVALUATION=1; shift ;;
    -h|--help)
      sed -n '1,28p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "${K}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --runs must be a positive integer" >&2
  exit 2
fi

cd "${REPO_ROOT}"
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export QUERY_CONCURRENCY="${QUERY_CONCURRENCY:-4}"
export QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1800}"

mkdir -p "${SPLIT_ROOT}" "${OUTPUT_ROOT}"

"${PYTHON}" - "${TRAIN_FILE}" "${EVAL_FILE}" "${SPLIT_ROOT}" <<'PY'
from pathlib import Path
import sys
from learning.heldout_runner import ALL_DATABASES, build_heldout_splits

train, evaluation, output = map(Path, sys.argv[1:])
for database in ALL_DATABASES:
    build_heldout_splits(train, evaluation, database, output)
PY

DATABASES=(california_schools european_football_2 formula_1 superhero)

latest_prompt() {
  local experiment_id="$1"
  local latest
  latest="$(find "${REPO_ROOT}/exp/learning/${experiment_id}" -mindepth 1 -maxdepth 1 \
    -type d -name '20*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${latest}" || ! -s "${latest}/experienced_prompt.md" ]]; then
    return 1
  fi
  printf '%s\n' "${latest}/experienced_prompt.md"
}

for database in "${DATABASES[@]}"; do
  experiment_id="cross_db_${database}"
  train_split="${SPLIT_ROOT}/train_${database}.jsonl"
  eval_split="${SPLIT_ROOT}/eval_${database}.jsonl"

  if [[ "${SKIP_TRAINING}" -eq 0 ]]; then
    "${PYTHON}" -m learning.cli \
      --config "${CONFIG}" \
      --exp_id "${experiment_id}" \
      --practice_path "${train_split}" \
      --eval_path "${eval_split}"
  fi

  prompt_file="$(latest_prompt "${experiment_id}")" || {
    echo "ERROR: no trained prompt for ${experiment_id}" >&2
    exit 3
  }

  if [[ "${SKIP_EVALUATION}" -eq 0 ]]; then
    for run_number in $(seq 1 "${K}"); do
      run_dir="${OUTPUT_ROOT}/${database}/run_${run_number}"
      if [[ -e "${run_dir}" ]]; then
        echo "ERROR: refusing to overwrite ${run_dir}" >&2
        exit 4
      fi
      mkdir -p "${run_dir}"
      "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
        --query_file "${eval_split}" \
        --prompt_file "${prompt_file}" \
        --out_dir "${run_dir}" \
        --concurrency "${QUERY_CONCURRENCY}" \
        --timeout_s "${QUERY_TIMEOUT_S}"
      "${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
        --run_dir "${run_dir}" \
        --query_file "${eval_split}" \
        --json > "${run_dir}/eval_summary.json"
    done
  fi
done

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${OUTPUT_ROOT}" \
  --agents "${DATABASES[@]}" \
  --k "${K}"
