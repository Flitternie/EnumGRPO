#!/usr/bin/env bash
#SBATCH --job-name=palimpzest-swan
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/palimpzest/slurm-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
PROGRAM_FILE="${REPO_ROOT}/baseline/palimpzest/programs/swan_oracle_efficiency.jsonl"
OUTPUT_ROOT="${PALIMPZEST_OUTPUT_ROOT:-${REPO_ROOT}/exp/baseline/palimpzest/oracle_efficiency}"
PALIMPZEST_SYSTEM_DIR="${PALIMPZEST_SYSTEM_DIR:-${REPO_ROOT}/baseline/palimpzest/system}"

export PALIMPZEST_SYSTEM_DIR
export QUERY_TIMEOUT_S="${QUERY_TIMEOUT_S:-1800}"
export QUERY_CONCURRENCY="${QUERY_CONCURRENCY:-4}"

[[ -s "${PROGRAM_FILE}" ]] || { echo "ERROR: missing ${PROGRAM_FILE}" >&2; exit 2; }
[[ -d "${PALIMPZEST_SYSTEM_DIR}" ]] || {
  echo "ERROR: Palimpzest checkout not found: ${PALIMPZEST_SYSTEM_DIR}" >&2
  echo "See baseline/palimpzest/README.md for the pinned upstream revision." >&2
  exit 2
}
EXPECTED_PALIMPZEST_COMMIT="807ed301c4d2457ef304647e6095184556bd1e83"
ACTUAL_PALIMPZEST_COMMIT="$(git -C "${PALIMPZEST_SYSTEM_DIR}" rev-parse HEAD 2>/dev/null || true)"
[[ "${ACTUAL_PALIMPZEST_COMMIT}" == "${EXPECTED_PALIMPZEST_COMMIT}" ]] || {
  echo "ERROR: expected Palimpzest ${EXPECTED_PALIMPZEST_COMMIT}, found ${ACTUAL_PALIMPZEST_COMMIT:-non-git checkout}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "ERROR: refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 3
}

cd "${REPO_ROOT}"
"${PYTHON}" baseline/palimpzest/run.py \
  --program_file "${PROGRAM_FILE}" \
  --out_dir "${OUTPUT_ROOT}" \
  --num_runs 3 \
  --query_timeout_s "${QUERY_TIMEOUT_S}" \
  --concurrency "${QUERY_CONCURRENCY}"

for run_number in 1 2 3; do
  "${PYTHON}" eval_swan.py \
    --run_dir "${OUTPUT_ROOT}/run_r${run_number}" \
    --query_file swan/evaluation.jsonl \
    --json > "${OUTPUT_ROOT}/run_r${run_number}/eval_summary.json"
done
