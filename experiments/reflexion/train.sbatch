#!/usr/bin/env bash
#SBATCH --job-name=reflexion-train
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/reflexion/slurm-train-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="${REPO_ROOT}/experiments/reflexion/config.yaml"

cd "${REPO_ROOT}"
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1

mkdir -p "${REPO_ROOT}/exp/learning/reflexion_swan40_n5_v1"
"${PYTHON}" -m learning.reflexion.cli --config "${CONFIG}"
