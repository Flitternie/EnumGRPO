#!/usr/bin/env bash
#SBATCH --job-name=bird-audit
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=exp/eval/pure_sql_bird_challenging_pilot_v1/logs/audit-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE_ROOT="${REPO_ROOT}/datasets/bird/raw/databases"
PREPARED_ROOT="${REPO_ROOT}/datasets/bird/prepared"
LOCAL_ROOT="${SLURM_TMPDIR:-$(mktemp -d /tmp/enumgrpo-bird-audit-${SLURM_JOB_ID}.XXXXXX)}"

mkdir -p "${REPO_ROOT}/exp/eval/pure_sql_bird_challenging_pilot_v1/logs"
mkdir -p "${LOCAL_ROOT}/sqlite" "${LOCAL_ROOT}/duckdb" "${LOCAL_ROOT}/out"
cp -a "${SOURCE_ROOT}/." "${LOCAL_ROOT}/sqlite/"
cp -a "${PREPARED_ROOT}/database/." "${LOCAL_ROOT}/duckdb/"

set +e
"${PYTHON}" "${REPO_ROOT}/experiments/cross_workload/bird/prepare_bird_duckdb.py" \
  --annotations "${REPO_ROOT}/datasets/bird/raw/annotations/dev_20251106.json" \
  --sqlite-root "${LOCAL_ROOT}/sqlite" \
  --duckdb-dir "${LOCAL_ROOT}/duckdb" \
  --output-jsonl "${LOCAL_ROOT}/out/evaluation_challenging_duckdb_agent.jsonl" \
  --output-report "${LOCAL_ROOT}/out/duckdb_challenging_report.json" \
  --difficulty challenging
status="$?"
set -e

cp -a "${LOCAL_ROOT}/out/." "${PREPARED_ROOT}/"
exit "${status}"
