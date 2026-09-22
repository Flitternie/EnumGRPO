#!/usr/bin/env bash
#SBATCH --job-name=vanilla-grpo-train
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=cpu-008-217,cpu-015-253
#SBATCH --output=experiments/vanilla_grpo/slurm-train-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ID="vanilla_grpo_swan40_n5_v1"
TRAIN_CONFIG="${REPO_ROOT}/experiments/vanilla_grpo/config.yaml"
TRAIN_ROOT="${REPO_ROOT}/exp/learning/${EXP_ID}"

cd "${REPO_ROOT}"

# Avoid inheriting stale tracing configuration that can make the installed
# OpenHands SDK fail during import before the experiment starts.
unset OTEL_ENDPOINT OTEL_EXPORTER_OTLP_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset LMNR_PROJECT_API_KEY
export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-python3}"

mkdir -p "${TRAIN_ROOT}"

echo "[$(date -Is)] Starting Vanilla GRPO training"
echo "Config: ${TRAIN_CONFIG}"
echo "Training output: ${TRAIN_ROOT}"

"${PYTHON}" -m learning.cli --config "${TRAIN_CONFIG}"
echo "[$(date -Is)] Vanilla GRPO training complete"
