#!/usr/bin/env bash
#SBATCH --job-name=spider-agent-smoke
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=exp/eval/pure_sql_spider/logs/smoke-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
GPT_ENV="${REPO_ROOT}/.env.gpt"
QUERY_FILE="${REPO_ROOT}/datasets/spider1/prepared/evaluation_extra_duckdb.jsonl"
DB_DIR="${REPO_ROOT}/datasets/spider1/prepared/database"
QUESTION_ID="${SPIDER_SMOKE_QID:-spider_dev_0024}"
OUT_DIR="${REPO_ROOT}/exp/eval/pure_sql_spider/smoke/job${SLURM_JOB_ID}"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300

mkdir -p "${OUT_DIR}"

echo "[$(date -Is)] Spider agent smoke: ${QUESTION_ID}"
echo "Output: ${OUT_DIR}"

"${DOTENV}" -f "${GPT_ENV}" run -- \
  "${PYTHON}" "${REPO_ROOT}/runners/agent.py" \
    --query_file "${QUERY_FILE}" \
    --db_dir "${DB_DIR}" \
    --out_dir "${OUT_DIR}" \
    --run_id "${QUESTION_ID}" \
    --concurrency 1 \
    --timeout_s 1800

echo "[$(date -Is)] Spider agent smoke complete"
