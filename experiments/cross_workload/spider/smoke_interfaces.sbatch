#!/usr/bin/env bash
#SBATCH --job-name=spider-iface-smoke
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=exp/eval/pure_sql_spider/logs/interface-smoke-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DOTENV="${DOTENV:-dotenv}"
OUT_DIR="${REPO_ROOT}/exp/eval/pure_sql_spider/smoke/interface_job${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
export PYTHONNOUSERSITE=1
export MCP_STARTUP_TIMEOUT_SEC=300
export RUN_SQL_FULL_CSV_EXPORT=1
export RUN_SQL_FULL_CSV_MAX_ROWS=1000

START="$(date +%s)"
timeout 90 "${PYTHON}" -c 'import duckdb, openai, pandas; import mcp.server.stdio; print("imports ready")'
END="$(date +%s)"
echo "[$(date -Is)] preflight=$((END - START))s"
if (( END - START > 30 )); then
  exit 75
fi

mkdir -p "${OUT_DIR}"
"${DOTENV}" -f "${REPO_ROOT}/.env" run -- \
  "${PYTHON}" "${REPO_ROOT}/run_swan_main.py" \
    --query_file "${REPO_ROOT}/datasets/spider1/prepared/evaluation_extra_duckdb_agent.jsonl" \
    --db_dir "${REPO_ROOT}/datasets/spider1/prepared/database" \
    --out_dir "${OUT_DIR}" \
    --prompt_file "${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md" \
    --run_id spider_dev_0024 \
    --concurrency 1 \
    --timeout_s 1800

test -s "${OUT_DIR}/spider_dev_0024.csv"
grep -q 'Official Spider schema metadata' "${OUT_DIR}/manifest.jsonl"
echo "[$(date -Is)] Spider interface smoke complete"
