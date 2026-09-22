#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${AXIS_ABLATION_OUTPUT_ROOT:-${REPO_ROOT}/exp/eval/experience_axis_ablation_swan_v1}"
MODEL_ENV="${AXIS_ABLATION_ENV_FILE:-${REPO_ROOT}/.env}"
SOURCE_POOL="${AXIS_ABLATION_SOURCE_POOL:-${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json}"
RUN_IDS="${AXIS_ABLATION_RUN_IDS:-1-3}"
CONDITION_TEXT="${AXIS_ABLATION_CONDITIONS:-without_execution_paradigm without_operator_type without_operator_placement without_selectivity_scope without_projection_width}"
read -r -a CONDITIONS <<< "${CONDITION_TEXT}"

[[ "${OUTPUT_ROOT}" == /* ]] || OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
[[ "${MODEL_ENV}" == /* ]] || MODEL_ENV="${REPO_ROOT}/${MODEL_ENV}"
[[ "${SOURCE_POOL}" == /* ]] || SOURCE_POOL="${REPO_ROOT}/${SOURCE_POOL}"

[[ -s "${MODEL_ENV}" ]] || { echo "ERROR: missing model environment: ${MODEL_ENV}" >&2; exit 2; }
[[ -s "${SOURCE_POOL}" ]] || { echo "ERROR: missing source pool: ${SOURCE_POOL}" >&2; exit 2; }

if [[ -e "${OUTPUT_ROOT}" && "${AXIS_ABLATION_ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: refusing to reuse existing output root: ${OUTPUT_ROOT}" >&2
  echo "Set AXIS_ABLATION_ALLOW_EXISTING=1 only to add new run IDs." >&2
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/metadata/pools"
cd "${REPO_ROOT}"

"${PYTHON}" experiments/axis_ablation/build_pools.py \
  --source-pool "${SOURCE_POOL}" \
  --axis-map experiments/axis_ablation/axis_map.json \
  --output-dir "${OUTPUT_ROOT}/metadata/pools" \
  > "${OUTPUT_ROOT}/logs/build_pools.log"

for condition in "${CONDITIONS[@]}"; do
  [[ -s "${OUTPUT_ROOT}/metadata/pools/${condition}.json" ]] || {
    echo "ERROR: unknown condition ${condition}" >&2
    exit 2
  }
  sbatch \
    --array="${RUN_IDS}" \
    --job-name="axis-${condition#without_}" \
    --output="${OUTPUT_ROOT}/logs/${condition}-%A_%a.log" \
    --export="ALL,AXIS_ABLATION_OUTPUT_ROOT=${OUTPUT_ROOT},AXIS_ABLATION_ENV_FILE=${MODEL_ENV},AXIS_ABLATION_CONDITION=${condition}" \
    experiments/axis_ablation/run_condition.sbatch
done
