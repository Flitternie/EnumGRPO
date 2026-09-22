#!/usr/bin/env bash
#SBATCH --job-name=axis-ablation
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DOTENV="${DOTENV:-dotenv}"
MODEL_ENV="${AXIS_ABLATION_ENV_FILE:?set AXIS_ABLATION_ENV_FILE}"
OUTPUT_ROOT="${AXIS_ABLATION_OUTPUT_ROOT:?set AXIS_ABLATION_OUTPUT_ROOT}"
CONDITION="${AXIS_ABLATION_CONDITION:?set AXIS_ABLATION_CONDITION}"
RUN_NUMBER="${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"
POOL_FILE="${OUTPUT_ROOT}/metadata/pools/${CONDITION}.json"
RUN_DIR="${OUTPUT_ROOT}/${CONDITION}/run_${RUN_NUMBER}"
QUERY_FILE="${REPO_ROOT}/swan/evaluation.jsonl"
SOURCE_DB_DIR="${REPO_ROOT}/swan/database"

cd "${REPO_ROOT}"
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC="${MCP_STARTUP_TIMEOUT_SEC:-300}"
export QUERY_CONCURRENCY="${QUERY_CONCURRENCY:-4}"
export QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1800}"

for required in "${MODEL_ENV}" "${QUERY_FILE}" "${POOL_FILE}"; do
  [[ -s "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 2; }
done
[[ -d "${SOURCE_DB_DIR}" ]] || { echo "ERROR: missing ${SOURCE_DB_DIR}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "ERROR: refusing to overwrite ${RUN_DIR}" >&2; exit 3; }

preflight_start="$(date +%s)"
timeout 90 "${PYTHON}" -c \
  'import duckdb, openai, pandas; import mcp.server.stdio; import litellm; import openhands.sdk'
preflight_elapsed="$(( $(date +%s) - preflight_start ))"
if (( preflight_elapsed > 30 )); then
  echo "ERROR: import preflight took ${preflight_elapsed}s" >&2
  exit 75
fi

LOCAL_SCRATCH="${SLURM_TMPDIR:-}"
REMOVE_LOCAL_SCRATCH=0
if [[ -z "${LOCAL_SCRATCH}" || ! -d "${LOCAL_SCRATCH}" ]]; then
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-axis-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
fi
cleanup() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-axis-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

LOCAL_DB_DIR="${LOCAL_SCRATCH}/swan_database"
mkdir -p "${LOCAL_DB_DIR}" "${RUN_DIR}"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_DB_DIR}/"

set +e
"${DOTENV}" -f "${MODEL_ENV}" run -- \
  "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
    --query_file "${QUERY_FILE}" \
    --db_dir "${LOCAL_DB_DIR}" \
    --out_dir "${RUN_DIR}" \
    --prompt_file "${POOL_FILE}" \
    --concurrency "${QUERY_CONCURRENCY}" \
    --timeout_s "${QUERY_TIMEOUT_S}"
runner_status="$?"
set -e

result_count=0
[[ ! -f "${RUN_DIR}/results.jsonl" ]] || result_count="$(wc -l < "${RUN_DIR}/results.jsonl")"
[[ "${result_count}" -eq 80 ]] || { echo "ERROR: produced ${result_count}/80 results" >&2; exit 4; }
[[ "${runner_status}" -eq 0 ]] || echo "Runner returned ${runner_status}; recorded task failures remain failures."

"${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
  --run_dir "${RUN_DIR}" \
  --query_file "${QUERY_FILE}" \
  --json > "${RUN_DIR}/eval_summary.json"
