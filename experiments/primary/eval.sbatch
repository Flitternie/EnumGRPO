#!/usr/bin/env bash
#SBATCH --job-name=enumgrpo-eval
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/primary/slurm-eval-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
POOL_FILE="${REPO_ROOT}/artifacts/experiences/swan_enumgrpo.json"
EVAL_ROOT="${ENUMGRPO_EVAL_ROOT:-${REPO_ROOT}/exp/eval/enumgrpo_swan40_n5_v1}"

cd "${REPO_ROOT}"
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export QUERY_CONCURRENCY="${QUERY_CONCURRENCY:-4}"
export QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1800}"

test -s "${POOL_FILE}"
LEARNED_AGENT_SLUG="enumgrpo" PYTHON="${PYTHON}" \
  bash "${REPO_ROOT}/run_multi_eval.sh" \
    -k 3 \
    -q "${REPO_ROOT}/swan/evaluation.jsonl" \
    -e "${POOL_FILE}" \
    -o "${EVAL_ROOT}"
