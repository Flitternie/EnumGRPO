#!/usr/bin/env bash
#SBATCH --job-name=lotus-swan
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/lotus/slurm-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
PROGRAM_FILE="${REPO_ROOT}/baseline/lotus/programs/swan_oracle_efficiency.jsonl"
OUTPUT_ROOT="${LOTUS_OUTPUT_ROOT:-${REPO_ROOT}/exp/baseline/lotus/oracle_efficiency}"
LOTUS_SYSTEM_DIR="${LOTUS_SYSTEM_DIR:-${REPO_ROOT}/baseline/lotus/system}"

export LOTUS_SYSTEM_DIR
export QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1800}"

[[ -s "${PROGRAM_FILE}" ]] || { echo "ERROR: missing ${PROGRAM_FILE}" >&2; exit 2; }
[[ -d "${LOTUS_SYSTEM_DIR}" ]] || {
  echo "ERROR: LOTUS checkout not found: ${LOTUS_SYSTEM_DIR}" >&2
  echo "See baseline/lotus/README.md for the pinned upstream revision." >&2
  exit 2
}
EXPECTED_LOTUS_COMMIT="136ae4f4a344a2f75d89f811e516dfcb0de30e46"
ACTUAL_LOTUS_COMMIT="$(git -C "${LOTUS_SYSTEM_DIR}" rev-parse HEAD 2>/dev/null || true)"
[[ "${ACTUAL_LOTUS_COMMIT}" == "${EXPECTED_LOTUS_COMMIT}" ]] || {
  echo "ERROR: expected LOTUS ${EXPECTED_LOTUS_COMMIT}, found ${ACTUAL_LOTUS_COMMIT:-non-git checkout}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "ERROR: refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 3
}

cd "${REPO_ROOT}"
"${PYTHON}" baseline/lotus/run.py \
  --program_file "${PROGRAM_FILE}" \
  --out_dir "${OUTPUT_ROOT}" \
  --num_runs 3 \
  --query_timeout_s "${QUERY_TIMEOUT_S}"

for run_number in 1 2 3; do
  "${PYTHON}" eval_swan.py \
    --run_dir "${OUTPUT_ROOT}/run_r${run_number}" \
    --query_file swan/evaluation.jsonl \
    --json > "${OUTPUT_ROOT}/run_r${run_number}/eval_summary.json"
done
