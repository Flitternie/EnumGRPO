#!/usr/bin/env bash
#SBATCH --job-name=axis-memory-ablation
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp/eval/experience_axis_ablation_swan_v1/logs/slurm-%j.log

set -euo pipefail

REPO_ROOT="/storage/home/lynie/projects/EnumGRPO"
PYTHON="/home/lynie/miniconda3/envs/enumgrpo-agent/bin/python"
DOTENV="/home/lynie/miniconda3/envs/enumgrpo-agent/bin/dotenv"
CLAUDE_ENV="${REPO_ROOT}/.env"
QUERY_FILE="${REPO_ROOT}/swan/evaluation.jsonl"
SOURCE_DB_DIR="${REPO_ROOT}/swan/database"
SOURCE_POOL="${REPO_ROOT}/exp/learning/main/experiences/latest.json"
AXIS_MAP="${REPO_ROOT}/experiments/axis_ablation/axis_map.json"
OUTPUT_ROOT="${AXIS_ABLATION_OUTPUT_ROOT:-${REPO_ROOT}/exp/eval/experience_axis_ablation_swan_v1}"
RUNS="${AXIS_ABLATION_RUNS:-3}"
POOL_DIR="${OUTPUT_ROOT}/metadata/pools"

CONDITIONS=(
  without_execution_paradigm
  without_operator_type
  without_operator_placement
  without_selectivity_scope
  without_projection_width
)

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300
export QUERY_CONCURRENCY=4
export QUERY_TIMEOUT_S=1800

if [[ ! "${RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: AXIS_ABLATION_RUNS must be a positive integer" >&2
  exit 2
fi
for required in "${PYTHON}" "${DOTENV}" "${CLAUDE_ENV}" "${QUERY_FILE}" "${SOURCE_POOL}" "${AXIS_MAP}"; do
  if [[ ! -s "${required}" ]]; then
    echo "ERROR: missing required file: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${SOURCE_DB_DIR}" ]]; then
  echo "ERROR: missing SWAN database directory: ${SOURCE_DB_DIR}" >&2
  exit 2
fi

for slug in "${CONDITIONS[@]}"; do
  if [[ -e "${OUTPUT_ROOT}/${slug}" ]]; then
    echo "ERROR: refusing to overwrite existing condition: ${OUTPUT_ROOT}/${slug}" >&2
    exit 3
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs" "${POOL_DIR}"
"${PYTHON}" "${REPO_ROOT}/experiments/axis_ablation/build_pools.py" \
  --source-pool "${SOURCE_POOL}" \
  --axis-map "${AXIS_MAP}" \
  --output-dir "${POOL_DIR}" \
  > "${OUTPUT_ROOT}/logs/build_pools.log"
cp -a "${AXIS_MAP}" "${OUTPUT_ROOT}/metadata/axis_map.json"

echo "[$(date -Is)] node=$(hostname) starting runtime preflight"
preflight_start="$(date +%s)"
timeout 90 "${PYTHON}" -c \
  'import duckdb, openai, pandas; import mcp.server.stdio; import litellm; import openhands.sdk; print("imports ready")'
preflight_end="$(date +%s)"
preflight_elapsed="$((preflight_end - preflight_start))"
echo "[$(date -Is)] import preflight completed in ${preflight_elapsed}s"
if (( preflight_elapsed > 30 )); then
  echo "ERROR: compute-node import preflight exceeded 30s; refusing to contaminate walltime" >&2
  exit 75
fi

LOCAL_SCRATCH="${SLURM_TMPDIR:-}"
REMOVE_LOCAL_SCRATCH=0
if [[ -z "${LOCAL_SCRATCH}" || ! -d "${LOCAL_SCRATCH}" ]]; then
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-axis-ablation-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
fi

cleanup_local_scratch() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-axis-ablation-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected scratch path ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup_local_scratch EXIT

LOCAL_DB_DIR="${LOCAL_SCRATCH}/swan_database"
mkdir -p "${LOCAL_DB_DIR}"
copy_start="$(date +%s)"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_DB_DIR}/"
copy_end="$(date +%s)"
echo "[$(date -Is)] staged SWAN databases locally in $((copy_end - copy_start))s"

run_condition() {
  local slug="$1"
  local pool_file="${POOL_DIR}/${slug}.json"
  local condition_root="${OUTPUT_ROOT}/${slug}"
  mkdir -p "${condition_root}"

  for run_number in $(seq 1 "${RUNS}"); do
    local run_dir="${condition_root}/run_${run_number}"
    local run_log="${OUTPUT_ROOT}/logs/${slug}-run_${run_number}.log"
    if [[ -e "${run_dir}" ]]; then
      echo "ERROR: refusing to overwrite existing run: ${run_dir}" >&2
      exit 3
    fi
    mkdir -p "${run_dir}"
    echo "[$(date -Is)] starting ${slug} run ${run_number}/${RUNS}"

    set +e
    "${DOTENV}" -f "${CLAUDE_ENV}" run -- \
      "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
        --query_file "${QUERY_FILE}" \
        --db_dir "${LOCAL_DB_DIR}" \
        --out_dir "${run_dir}" \
        --prompt_file "${pool_file}" \
        --concurrency 4 \
        --timeout_s 1800 \
      2>&1 | tee -a "${run_log}"
    runner_status="${PIPESTATUS[0]}"
    set -e

    result_count=0
    if [[ -f "${run_dir}/results.jsonl" ]]; then
      result_count="$(wc -l < "${run_dir}/results.jsonl")"
    fi
    if [[ "${result_count}" -ne 80 ]]; then
      echo "ERROR: ${slug} run ${run_number} produced ${result_count}/80 result records" >&2
      exit 4
    fi
    if [[ "${runner_status}" -ne 0 ]]; then
      echo "[$(date -Is)] runner returned ${runner_status}; recorded task failures remain failures"
    fi

    "${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
      --run_dir "${run_dir}" \
      --query_file "${QUERY_FILE}" \
      --json > "${run_dir}/eval_summary.json"
    echo "[$(date -Is)] completed ${slug} run ${run_number}/${RUNS}"
  done
}

for condition in "${CONDITIONS[@]}"; do
  run_condition "${condition}"
done

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${OUTPUT_ROOT}" \
  --agents "${CONDITIONS[@]}" \
  --k "${RUNS}" \
  > "${OUTPUT_ROOT}/aggregate_table.md"

echo "[$(date -Is)] experience-axis leave-one-out ablation complete"
