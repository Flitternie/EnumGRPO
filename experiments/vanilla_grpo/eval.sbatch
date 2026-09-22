#!/usr/bin/env bash
#SBATCH --job-name=vanilla-grpo-eval
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/vanilla_grpo/slurm-eval-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
EXP_ID="vanilla_grpo_swan40_n5_v1"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_vanilla_grpo.json"
EVAL_ROOT="${REPO_ROOT}/exp/eval/${EXP_ID}"

cd "${REPO_ROOT}"

unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
# Login-shell proxy endpoints are local to the login node and refuse
# connections from compute nodes.  Compute nodes can reach the Llama API
# directly, so prevent HTTP clients from inheriting those stale proxy values.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1

# Pin the evaluation settings used by the existing comparison runs.
export QUERY_CONCURRENCY=4
export QUERY_TIMEOUT_S=1800

if [[ ! -s "${POOL_FILE}" ]]; then
  echo "ERROR: Vanilla GRPO experience pool not found: ${POOL_FILE}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"

echo "[$(date -Is)] Starting clean Vanilla GRPO evaluation"
echo "Pool: ${POOL_FILE}"
echo "Output: ${EVAL_ROOT}"

LEARNED_AGENT_SLUG="vanilla_grpo" \
PYTHON="${PYTHON}" \
bash "${REPO_ROOT}/run_multi_eval.sh" \
  -k 3 \
  -q "${REPO_ROOT}/swan/evaluation.jsonl" \
  -e "${POOL_FILE}" \
  -o "${EVAL_ROOT}"

echo "[$(date -Is)] Vanilla GRPO evaluation complete"
