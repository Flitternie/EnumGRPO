#!/usr/bin/env bash
#SBATCH --job-name=reflexion-eval
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/reflexion/slurm-eval-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_reflexion.json"
EVAL_ROOT="${REPO_ROOT}/exp/eval/reflexion_swan40_n5_v1"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1

export QUERY_CONCURRENCY=4
export QUERY_TIMEOUT_S=1800

if [[ ! -s "${POOL_FILE}" ]]; then
  echo "ERROR: Reflexion experience pool not found: ${POOL_FILE}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"

echo "[$(date -Is)] Starting Reflexion-style K=3 evaluation"
echo "Pool: ${POOL_FILE}"
echo "Output: ${EVAL_ROOT}"

LEARNED_AGENT_SLUG="reflexion" \
PYTHON="${PYTHON}" \
bash "${REPO_ROOT}/run_multi_eval.sh" \
  -k 3 \
  -q "${REPO_ROOT}/swan/evaluation.jsonl" \
  -e "${POOL_FILE}" \
  -o "${EVAL_ROOT}"

echo "[$(date -Is)] Reflexion-style evaluation complete"
