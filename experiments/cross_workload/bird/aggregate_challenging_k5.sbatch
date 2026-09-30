#!/usr/bin/env bash
#SBATCH --job-name=bird-pure-sql-agg-k5
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=exp/eval/pure_sql_bird_challenging/logs/aggregate-k5-slurm-%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${REPO_ROOT}/exp/eval/pure_sql_bird_challenging"

cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1

for condition in agent_no_pool agent_with_swan_pool; do
  for run_index in 1 2 3 4 5; do
    run_dir="${OUTPUT_ROOT}/${condition}/run_${run_index}"
    count="$(wc -l < "${run_dir}/results.jsonl" 2>/dev/null || echo 0)"
    if [[ "${count}" -ne 231 || ! -s "${run_dir}/eval_summary.json" ]]; then
      echo "ERROR: incomplete ${condition}/run_${run_index}: ${count}/231 results" >&2
      exit 1
    fi
  done
done

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
  --base_dir "${OUTPUT_ROOT}" \
  --agents agent_no_pool agent_with_swan_pool \
  --k 5

echo "[$(date -Is)] BIRD five-run aggregation complete"
