#!/usr/bin/env bash
#SBATCH --job-name=bird-pure-sql-r1
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp/eval/pure_sql_bird_challenging/logs/run1-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
CLAUDE_ENV="${REPO_ROOT}/.env"
QUERY_FILE="${REPO_ROOT}/datasets/bird/prepared/evaluation_challenging_duckdb_agent.jsonl"
SOURCE_DB_DIR="${REPO_ROOT}/datasets/bird/prepared/database"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json"
PLAIN_PROMPT="${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md"
OUTPUT_ROOT="${REPO_ROOT}/exp/eval/pure_sql_bird_challenging"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300
export QUERY_CONCURRENCY=4
export QUERY_TIMEOUT_S=1800
export RUN_SQL_FULL_CSV_EXPORT=1
export RUN_SQL_FULL_CSV_MAX_ROWS=100000

if [[ ! -s "${CLAUDE_ENV}" || ! -s "${QUERY_FILE}" || ! -s "${POOL_FILE}" ]]; then
  echo "ERROR: missing Claude env, audited BIRD query file, or frozen pool" >&2
  exit 1
fi
if [[ "$(wc -l < "${QUERY_FILE}")" -ne 231 ]]; then
  echo "ERROR: audited BIRD query file does not contain 231 records" >&2
  exit 1
fi
for target in \
  "${OUTPUT_ROOT}/agent_no_pool/run_1" \
  "${OUTPUT_ROOT}/agent_with_swan_pool/run_1"; do
  if [[ -e "${target}" ]]; then
    echo "ERROR: refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

echo "[$(date -Is)] node=$(hostname) starting BIRD runtime preflight"
START="$(date +%s)"
timeout 90 "${PYTHON}" -c \
  'import duckdb, openai, pandas; import mcp.server.stdio; import litellm; import openhands.sdk; print("imports ready")'
END="$(date +%s)"
ELAPSED="$((END - START))"
echo "[$(date -Is)] import preflight completed in ${ELAPSED}s"
if (( ELAPSED > 30 )); then
  echo "ERROR: compute-node import preflight exceeded 30s" >&2
  exit 75
fi

LOCAL_SCRATCH="${SLURM_TMPDIR:-}"
REMOVE_LOCAL_SCRATCH=0
if [[ -z "${LOCAL_SCRATCH}" || ! -d "${LOCAL_SCRATCH}" ]]; then
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-bird-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
fi
cleanup_local_scratch() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-bird-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected scratch path ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup_local_scratch EXIT

LOCAL_DB_DIR="${LOCAL_SCRATCH}/bird_database"
mkdir -p "${LOCAL_DB_DIR}"
COPY_START="$(date +%s)"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_DB_DIR}/"
COPY_END="$(date +%s)"
echo "[$(date -Is)] staged BIRD databases to local scratch in $((COPY_END - COPY_START))s"

run_condition() {
  local slug="$1"
  local prompt_file="$2"
  local run_dir="${OUTPUT_ROOT}/${slug}/run_1"
  mkdir -p "${run_dir}"
  echo "[$(date -Is)] starting ${slug} run 1/1"

  set +e
  "${DOTENV}" -f "${CLAUDE_ENV}" run -- \
    "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
      --query_file "${QUERY_FILE}" \
      --db_dir "${LOCAL_DB_DIR}" \
      --out_dir "${run_dir}" \
      --prompt_file "${prompt_file}" \
      --concurrency 4 \
      --timeout_s 1800
  runner_status="$?"
  set -e

  result_count="$(wc -l < "${run_dir}/results.jsonl" 2>/dev/null || echo 0)"
  if [[ "${result_count}" -ne 231 ]]; then
    echo "ERROR: ${slug} produced ${result_count}/231 result records" >&2
    exit 1
  fi
  if [[ "${runner_status}" -ne 0 ]]; then
    echo "[$(date -Is)] ${slug} had task failures; retaining them as failures"
  fi

  "${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
    --run_dir "${run_dir}" \
    --query_file "${QUERY_FILE}" \
    --json > "${run_dir}/eval_summary.json"
  echo "[$(date -Is)] completed ${slug} run 1/1"
}

run_condition "agent_no_pool" "${PLAIN_PROMPT}"
run_condition "agent_with_swan_pool" "${POOL_FILE}"

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${OUTPUT_ROOT}" \
  --agents agent_no_pool agent_with_swan_pool \
  --k 1

echo "[$(date -Is)] BIRD no-pool/with-pool run 1 complete"
