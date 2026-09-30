#!/usr/bin/env bash
#SBATCH --job-name=sembench-blendsql
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=exp/eval/sembench_movie/logs/blendsql-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${SEMBENCH_OUTPUT_DIR:-${REPO_ROOT}/exp/eval/sembench_movie/baselines_same_llm/blendsql/run_1}"
PREPARED_ROOT="${REPO_ROOT}/datasets/sembench/prepared/movie_sf2000"
QUERIES_DIR="${REPO_ROOT}/experiments/cross_workload/sembench/oracle_programs/blendsql"

cd "${REPO_ROOT}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv-sembench-blendsql"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/exp/eval/sembench_movie/logs"

LOCAL_SCRATCH="${SLURM_TMPDIR:-}"
REMOVE_LOCAL_SCRATCH=0
if [[ -z "${LOCAL_SCRATCH}" || ! -d "${LOCAL_SCRATCH}" ]]; then
  LOCAL_SCRATCH="$(mktemp -d "/tmp/enumgrpo-sembench-blendsql-${SLURM_JOB_ID}.XXXXXX")"
  REMOVE_LOCAL_SCRATCH=1
fi
cleanup_local_scratch() {
  if [[ "${REMOVE_LOCAL_SCRATCH}" -eq 1 ]]; then
    case "${LOCAL_SCRATCH}" in
      "/tmp/enumgrpo-sembench-blendsql-${SLURM_JOB_ID}."*) rm -rf -- "${LOCAL_SCRATCH}" ;;
      *) echo "WARNING: refusing to remove unexpected scratch path ${LOCAL_SCRATCH}" >&2 ;;
    esac
  fi
}
trap cleanup_local_scratch EXIT

LOCAL_DB="${LOCAL_SCRATCH}/sembench_movie.duckdb"
cp -a "${PREPARED_ROOT}/database/sembench_movie.duckdb" "${LOCAL_DB}"

set +e
uv run --python 3.12 --with blendsql==0.1.26 --with python-dotenv \
  "${REPO_ROOT}/experiments/cross_workload/sembench/run_blendsql.py" \
  --db "${LOCAL_DB}" \
  --queries-dir "${QUERIES_DIR}" \
  --out-dir "${OUTPUT_DIR}" \
  --env-file "${REPO_ROOT}/.env" \
  --concurrency 32
runner_status="$?"
set -e

"${PYTHON}" "${REPO_ROOT}/experiments/cross_workload/sembench/evaluate_movie.py" \
  --run-dir "${OUTPUT_DIR}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output "${OUTPUT_DIR}/eval_summary.json"

exit "${runner_status}"
