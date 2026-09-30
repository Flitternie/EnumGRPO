#!/usr/bin/env bash
#SBATCH --job-name=spider-r4-r5
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp/eval/pure_sql_spider/logs/extra-r4-r5-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
CLAUDE_ENV="${REPO_ROOT}/.env"
QUERY_FILE="${REPO_ROOT}/datasets/spider1/prepared/evaluation_extra_duckdb_agent.jsonl"
SOURCE_DB_DIR="${REPO_ROOT}/datasets/spider1/prepared/database"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json"
PLAIN_PROMPT="${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md"
OUTPUT_ROOT="${REPO_ROOT}/exp/eval/pure_sql_spider"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300
export QUERY_CONCURRENCY=4
export QUERY_TIMEOUT_S=1800
export RUN_SQL_FULL_CSV_EXPORT=1
export RUN_SQL_FULL_CSV_MAX_ROWS=1000

if [[ ! -s "${CLAUDE_ENV}" || ! -s "${QUERY_FILE}" || ! -s "${POOL_FILE}" ]]; then
  echo "ERROR: missing Claude env, Spider query file, or frozen pool" >&2
  exit 1
fi

# Check all targets before creating any of them.  Existing runs are never
# removed or reused by this append-only job.
for target in \
  "${OUTPUT_ROOT}/agent_no_pool/run_4" \
  "${OUTPUT_ROOT}/agent_with_swan_pool/run_4" \
  "${OUTPUT_ROOT}/agent_no_pool/run_5" \
  "${OUTPUT_ROOT}/agent_with_swan_pool/run_5"; do
  if [[ -e "${target}" ]]; then
    echo "ERROR: refusing to overwrite existing ${target}" >&2
    exit 1
  fi
done

echo "[$(date -Is)] node=$(hostname) starting Spider runtime preflight"
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
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-spider-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
  echo "[$(date -Is)] SLURM_TMPDIR unavailable; using ${LOCAL_SCRATCH}"
fi

cleanup_local_scratch() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-spider-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected scratch path ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup_local_scratch EXIT

LOCAL_DB_DIR="${LOCAL_SCRATCH}/spider_database"
mkdir -p "${LOCAL_DB_DIR}"
COPY_START="$(date +%s)"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_DB_DIR}/"
COPY_END="$(date +%s)"
echo "[$(date -Is)] staged Spider databases to local scratch in $((COPY_END - COPY_START))s"

run_one() {
  local slug="$1"
  local run_number="$2"
  local prompt_file="$3"
  local run_dir="${OUTPUT_ROOT}/${slug}/run_${run_number}"

  if [[ -e "${run_dir}" ]]; then
    echo "ERROR: refusing to overwrite existing ${run_dir}" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  echo "[$(date -Is)] starting ${slug} run ${run_number}"

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
  if [[ "${result_count}" -ne 166 ]]; then
    echo "ERROR: ${slug} run ${run_number} produced ${result_count}/166 result records" >&2
    exit 1
  fi
  if [[ "${runner_status}" -ne 0 ]]; then
    echo "[$(date -Is)] ${slug} run ${run_number} had task failures; retaining them as failures"
  fi

  "${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
    --run_dir "${run_dir}" \
    --query_file "${QUERY_FILE}" \
    --json > "${run_dir}/eval_summary.json"
  echo "[$(date -Is)] completed ${slug} run ${run_number}"
}

run_one "agent_no_pool" 4 "${PLAIN_PROMPT}"
run_one "agent_with_swan_pool" 4 "${POOL_FILE}"
run_one "agent_no_pool" 5 "${PLAIN_PROMPT}"
run_one "agent_with_swan_pool" 5 "${POOL_FILE}"

"${PYTHON}" "${REPO_ROOT}/experiments/cross_workload/spider/aggregate_spider_k5.py"

echo "[$(date -Is)] Spider appended runs 4-5 and K=5 aggregation complete"
