#!/usr/bin/env bash
#SBATCH --job-name=sembench-movie
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=exp/eval/sembench_movie/logs/formal-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
CLAUDE_ENV="${REPO_ROOT}/.env"
PREPARED="${REPO_ROOT}/datasets/sembench/prepared/movie_sf2000"
QUERY_FILE="${PREPARED}/evaluation_movie_agent.jsonl"
SOURCE_DB_DIR="${PREPARED}/database"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json"
PLAIN_PROMPT="${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md"
OUTPUT_ROOT="${REPO_ROOT}/exp/eval/sembench_movie"
RUNS="${SEMBENCH_RUNS:-1}"
RUN_START="${SEMBENCH_RUN_START:-1}"
CONDITIONS="${SEMBENCH_CONDITIONS:-all}"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300
export QUERY_CONCURRENCY=2
export QUERY_TIMEOUT_S=1800
export AGENT_CLAUDE_THINKING=enabled
export AGENT_CLAUDE_REQUEST_STYLE=compat
export AGENT_MAX_ITERATIONS=30
export RUN_SQL_FULL_CSV_EXPORT=1
export RUN_SQL_FULL_CSV_MAX_ROWS=100000

if [[ ! -s "${CLAUDE_ENV}" || ! -s "${QUERY_FILE}" || ! -s "${POOL_FILE}" ]]; then
  echo "ERROR: missing model env, prepared SemBench queries, or SWAN experience file" >&2
  exit 1
fi
if [[ "$(wc -l < "${QUERY_FILE}")" -ne 10 ]]; then
  echo "ERROR: expected all 10 Movie queries" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

echo "[$(date -Is)] node=$(hostname) starting runtime preflight"
start="$(date +%s)"
timeout 90 "${PYTHON}" -c \
  'import duckdb, openai, pandas; import mcp.server.stdio; import litellm; import openhands.sdk; print("imports ready")'
elapsed="$(( $(date +%s) - start ))"
if (( elapsed > 30 )); then
  echo "ERROR: compute-node import preflight exceeded 30s" >&2
  exit 75
fi

LOCAL_SCRATCH="${SLURM_TMPDIR:-}"
REMOVE_LOCAL_SCRATCH=0
if [[ -z "${LOCAL_SCRATCH}" || ! -d "${LOCAL_SCRATCH}" ]]; then
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-sembench-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
fi
cleanup_local_scratch() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-sembench-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected scratch path ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup_local_scratch EXIT

LOCAL_DB_DIR="${LOCAL_SCRATCH}/sembench_movie_database"
mkdir -p "${LOCAL_DB_DIR}"
cp -a "${SOURCE_DB_DIR}/." "${LOCAL_DB_DIR}/"

evaluate_run() {
  local run_dir="$1"
  "${PYTHON}" "${REPO_ROOT}/experiments/cross_workload/sembench/evaluate_movie.py" \
    --run-dir "${run_dir}" \
    --prepared-root "${PREPARED}" \
    --output "${run_dir}/eval_summary.json"
}

run_agent_condition() {
  local slug="$1"
  local prompt_file="$2"
  local run_end="$((RUN_START + RUNS - 1))"
  for run_number in $(seq "${RUN_START}" "${run_end}"); do
    local run_dir="${OUTPUT_ROOT}/${slug}/run_${run_number}"
    if [[ -e "${run_dir}" ]]; then
      echo "ERROR: refusing to overwrite ${run_dir}" >&2
      exit 1
    fi
    mkdir -p "${run_dir}"
    set +e
    "${DOTENV}" -f "${CLAUDE_ENV}" run -- \
      "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
        --query_file "${QUERY_FILE}" \
        --db_dir "${LOCAL_DB_DIR}" \
        --out_dir "${run_dir}" \
        --prompt_file "${prompt_file}" \
        --concurrency 2 \
        --timeout_s 1800
    runner_status="$?"
    set -e
    if [[ "$(wc -l < "${run_dir}/results.jsonl" 2>/dev/null || echo 0)" -ne 10 ]]; then
      echo "ERROR: ${slug} run ${run_number} did not record all 10 tasks" >&2
      exit 1
    fi
    if [[ "${runner_status}" -ne 0 ]]; then
      echo "WARNING: retaining task failures for evaluation"
    fi
    evaluate_run "${run_dir}"
  done
}

run_text2sql_condition() {
  local run_end="$((RUN_START + RUNS - 1))"
  for run_number in $(seq "${RUN_START}" "${run_end}"); do
    local run_dir="${OUTPUT_ROOT}/agentic_text2sql/run_${run_number}"
    if [[ -e "${run_dir}" ]]; then
      echo "ERROR: refusing to overwrite ${run_dir}" >&2
      exit 1
    fi
    mkdir -p "${run_dir}"
    set +e
    "${DOTENV}" -f "${CLAUDE_ENV}" run -- \
      "${PYTHON}" "${REPO_ROOT}/run_swan_agentic_text2sql.py" \
        --query_file "${QUERY_FILE}" \
        --db_dir "${LOCAL_DB_DIR}" \
        --out_dir "${run_dir}" \
        --concurrency 2 \
        --timeout_s 1800
    runner_status="$?"
    set -e
    if [[ "$(wc -l < "${run_dir}/results.jsonl" 2>/dev/null || echo 0)" -ne 10 ]]; then
      echo "ERROR: Agentic Text2SQL run ${run_number} did not record all 10 tasks" >&2
      exit 1
    fi
    if [[ "${runner_status}" -ne 0 ]]; then
      echo "WARNING: retaining task failures for evaluation"
    fi
    evaluate_run "${run_dir}"
  done
}

case "${CONDITIONS}" in
  all)
    run_text2sql_condition
    run_agent_condition "agent_no_pool" "${PLAIN_PROMPT}"
    run_agent_condition "agent_with_swan_experiences" "${POOL_FILE}"
    ;;
  agents)
    run_agent_condition "agent_no_pool" "${PLAIN_PROMPT}"
    run_agent_condition "agent_with_swan_experiences" "${POOL_FILE}"
    ;;
  text2sql)
    run_text2sql_condition
    ;;
  no_pool)
    run_agent_condition "agent_no_pool" "${PLAIN_PROMPT}"
    ;;
  with_pool)
    run_agent_condition "agent_with_swan_experiences" "${POOL_FILE}"
    ;;
  *)
    echo "ERROR: SEMBENCH_CONDITIONS must be all, agents, text2sql, no_pool, or with_pool" >&2
    exit 2
    ;;
esac

if [[ -s "${OUTPUT_ROOT}/agentic_text2sql/run_${RUNS}/eval_summary.json" && \
      -s "${OUTPUT_ROOT}/agent_no_pool/run_${RUNS}/eval_summary.json" && \
      -s "${OUTPUT_ROOT}/agent_with_swan_experiences/run_${RUNS}/eval_summary.json" ]]; then
  "${PYTHON}" "${REPO_ROOT}/experiments/cross_workload/sembench/aggregate_movie.py" \
    --experiment-root "${OUTPUT_ROOT}" \
    --runs "${RUNS}" \
    --output "${OUTPUT_ROOT}/aggregate_summary.json"
fi

echo "[$(date -Is)] SemBench Movie experiment complete"
