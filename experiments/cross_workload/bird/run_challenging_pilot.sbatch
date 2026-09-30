#!/usr/bin/env bash
#SBATCH --job-name=bird-pilot
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp/eval/pure_sql_bird_challenging_pilot_v1/logs/eval-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
QUERY_FILE="${REPO_ROOT}/datasets/bird/prepared/evaluation_challenging_duckdb_agent.jsonl"
SOURCE_DB_DIR="${REPO_ROOT}/datasets/bird/prepared/database"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json"
PLAIN_PROMPT="${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md"
OUTPUT_ROOT="${REPO_ROOT}/exp/eval/pure_sql_bird_challenging_pilot_v1"

cd "${REPO_ROOT}"
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1 MCP_STARTUP_TIMEOUT_SEC=300
export QUERY_CONCURRENCY=4 QUERY_TIMEOUT_S=1800
export RUN_SQL_FULL_CSV_EXPORT=1 RUN_SQL_FULL_CSV_MAX_ROWS=100000

LOCAL_ROOT="${SLURM_TMPDIR:-$(mktemp -d /tmp/enumgrpo-bird-${SLURM_JOB_ID}.XXXXXX)}"
mkdir -p "${LOCAL_ROOT}/database"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_ROOT}/database/"

run_condition() {
  local slug="$1" prompt="$2" run_dir="${OUTPUT_ROOT}/$1/run_1"
  if [[ -e "${run_dir}" ]]; then
    echo "ERROR: refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  set +e
  "${DOTENV}" -f "${REPO_ROOT}/.env" run -- \
    "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
      --query_file "${QUERY_FILE}" --db_dir "${LOCAL_ROOT}/database" \
      --out_dir "${run_dir}" --prompt_file "${prompt}" \
      --concurrency 4 --timeout_s 1800
  status="$?"
  set -e
  count="$(wc -l < "${run_dir}/results.jsonl" 2>/dev/null || echo 0)"
  [[ "${count}" -eq 231 ]] || { echo "ERROR: ${slug} has ${count}/231 records" >&2; exit 1; }
  "${PYTHON}" "${REPO_ROOT}/eval_swan.py" \
    --run_dir "${run_dir}" --query_file "${QUERY_FILE}" --json \
    > "${run_dir}/eval_summary.json"
  [[ "${status}" -eq 0 ]] || echo "${slug}: task failures retained as failures"
}

run_condition agent_no_pool "${PLAIN_PROMPT}"
run_condition agent_with_swan_pool "${POOL_FILE}"

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${OUTPUT_ROOT}" \
  --agents agent_no_pool agent_with_swan_pool --k 1
