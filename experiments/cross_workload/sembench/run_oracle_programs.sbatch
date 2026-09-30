#!/usr/bin/env bash
#SBATCH --job-name=sembench-baseline
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SYSTEM="${SEMBENCH_SYSTEM:?Set SEMBENCH_SYSTEM to lotus or palimpzest}"
QUERIES="${SEMBENCH_QUERIES:-1}"
OUTPUT_DIR="${SEMBENCH_OUTPUT_DIR:?Set SEMBENCH_OUTPUT_DIR}"
EVAL_OUTPUT="${SEMBENCH_EVAL_OUTPUT:-}"
PREPARED_ROOT="${SEMBENCH_PREPARED_ROOT:-datasets/sembench/prepared/movie_sf2000}"

case "${SYSTEM}" in
  lotus) PYTHON="${REPO_ROOT}/baseline/lotus/.venv/bin/python" ;;
  palimpzest) PYTHON="${REPO_ROOT}/baseline/palimpzest/.venv/bin/python" ;;
  *) echo "ERROR: unsupported system ${SYSTEM}" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1

"${PYTHON}" experiments/cross_workload/sembench/run_oracle_programs.py \
  --system "${SYSTEM}" \
  --sembench-root datasets/sembench/source \
  --queries "${QUERIES}" \
  --out-dir "${OUTPUT_DIR}"

if [[ -n "${EVAL_OUTPUT}" ]]; then
  "${PYTHON}" experiments/cross_workload/sembench/evaluate_movie.py \
    --run-dir "${OUTPUT_DIR}" \
    --prepared-root "${PREPARED_ROOT}" \
    --output "${EVAL_OUTPUT}"
fi
