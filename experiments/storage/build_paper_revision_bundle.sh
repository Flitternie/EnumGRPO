#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/storage/home/lynie/projects/EnumGRPO}"
BUNDLE_NAME="${2:-paper_revision_experiments_20260914}"
ARCHIVES_DIR="${ROOT}/archives"
STAGING="${ARCHIVES_DIR}/${BUNDLE_NAME}"
ARCHIVE="${ARCHIVES_DIR}/${BUNDLE_NAME}.tar.zst"
PARTIAL="${ARCHIVE}.partial"

if [[ ! -d "${ROOT}/experiments" || ! -d "${ROOT}/exp" ]]; then
  echo "Invalid repository root: ${ROOT}" >&2
  exit 2
fi

for target in "${STAGING}" "${ARCHIVE}" "${PARTIAL}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing target: ${target}" >&2
    exit 3
  fi
done

mkdir -p "${ARCHIVES_DIR}" "${STAGING}"
cp -a "${ROOT}/experiments/storage/paper_revision_bundle_README.md" "${STAGING}/README.md"

copy_file() {
  local source_rel="$1"
  local destination_rel="$2"
  local source="${ROOT}/${source_rel}"
  local destination="${STAGING}/${destination_rel}"
  [[ -f "${source}" ]] || { echo "Missing file: ${source}" >&2; exit 4; }
  mkdir -p "$(dirname "${destination}")"
  cp -al "${source}" "${destination}"
}

copy_tree() {
  local source_rel="$1"
  local destination_rel="$2"
  shift 2
  local source="${ROOT}/${source_rel}/"
  local destination="${STAGING}/${destination_rel}/"
  [[ -d "${source}" ]] || { echo "Missing directory: ${source}" >&2; exit 4; }
  mkdir -p "${destination}"
  rsync -a --link-dest="${ROOT}/${source_rel}" \
    --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
    --exclude='.cache/' --exclude='runs/' \
    --exclude='*.pyc' --exclude='.env' --exclude='.env.*' \
    "$@" "${source}" "${destination}"
}

copy_common_runtime() {
  local setup="$1"
  copy_tree agent "${setup}/code/agent"
  copy_tree learning "${setup}/code/learning"
  copy_tree tools "${setup}/code/tools"
  copy_tree utils "${setup}/code/utils"
  for path in aggregate_runs.py eval_swan.py mcp_server.py run_multi_eval.sh run_swan_main.py environment.yml example.env; do
    copy_file "${path}" "${setup}/code/${path}"
  done
}

# 1. Oracle-assisted LOTUS.
setup="lotus_oracle_assisted"
copy_tree baseline/lotus "${setup}/code" \
  --exclude='system/' --exclude='runs/' --exclude='legacy_runs/' \
  --exclude='generated_queries_*.jsonl' --exclude='generated_queries_*.manifest.json'
copy_file baseline/lotus/generated_queries_oracle_efficiency.jsonl "${setup}/inputs/generated_queries_oracle_efficiency.jsonl"
copy_file baseline/lotus/generated_queries_oracle_efficiency.jsonl.manifest.json "${setup}/inputs/generated_queries_oracle_efficiency.jsonl.manifest.json"
copy_tree exp/baseline/lotus/oracle_efficiency_k3_1800s_physical_c32 "${setup}/results"
copy_file environment.yml "${setup}/code/environment.yml"

# 2. Oracle-assisted Palimpzest.
setup="palimpzest_oracle_assisted"
copy_tree baseline/palimpzest "${setup}/code" \
  --exclude='system/' --exclude='runs/' --exclude='legacy_exp/' \
  --exclude='generated_queries_*.jsonl' --exclude='generated_queries_*.manifest.json'
copy_file baseline/palimpzest/generated_queries_oracle_efficiency.jsonl "${setup}/inputs/generated_queries_oracle_efficiency.jsonl"
copy_file baseline/palimpzest/generated_queries_oracle_efficiency.jsonl.manifest.json "${setup}/inputs/generated_queries_oracle_efficiency.jsonl.manifest.json"
copy_tree exp/baseline/palimpzest/oracle_efficiency_k3_1800s_physical_c4 "${setup}/results"
copy_file environment.yml "${setup}/code/environment.yml"

# 3. Vanilla GRPO / no plan enumeration.
setup="vanilla_grpo"
copy_common_runtime "${setup}"
copy_tree experiments/vanilla_grpo "${setup}/code/experiment"
copy_file swan/learning.jsonl "${setup}/inputs/swan_learning.jsonl"
copy_file swan/evaluation.jsonl "${setup}/inputs/swan_evaluation.jsonl"
copy_file artifacts/experiences/swan_vanilla_grpo.json "${setup}/inputs/experience_pool.json"
copy_tree exp/learning/vanilla_grpo_swan40_n5_v1 "${setup}/results/learning"
copy_tree exp/eval/vanilla_grpo_swan40_n5_v1 "${setup}/results/evaluation"

# 4. Reflexion-style independent reflection.
setup="reflexion"
copy_common_runtime "${setup}"
copy_tree experiments/reflexion "${setup}/code/experiment"
copy_file swan/evaluation.jsonl "${setup}/inputs/swan_evaluation.jsonl"
copy_file artifacts/experiences/swan_reflexion.json "${setup}/inputs/experience_pool.json"
copy_tree exp/eval/reflexion_swan40_n5_v1 "${setup}/results/evaluation"

# 5. GPT backbone transfer.
setup="gpt_backbone_transfer"
copy_common_runtime "${setup}"
copy_tree experiments/gpt_transfer "${setup}/code/experiment" \
  --exclude='repair_smoke*' --exclude='postprocess_with_pool.sbatch' \
  --exclude='resume_with_pool_runs_2_3.sbatch' --exclude='resume_with_pool_runs_2_3.sh' \
  --exclude='run_extra_runs_*'
copy_file swan/evaluation.jsonl "${setup}/inputs/swan_evaluation.jsonl"
copy_file artifacts/experiences/swan_enumgrpo.json "${setup}/inputs/experience_pool.json"
copy_tree exp/eval/gpt_transfer "${setup}/results"

# 6. Spider pure-SQL transfer.
setup="pure_sql_spider"
copy_common_runtime "${setup}"
copy_tree experiments/spider_pure_sql "${setup}/code/experiment" \
  --exclude='smoke_*' --exclude='run_extra_runs_*'
for path in evaluation_extra_duckdb.jsonl evaluation_extra_duckdb_agent.jsonl duckdb_extra_report.json sample_manifest.json; do
  copy_file "datasets/spider1/prepared/${path}" "${setup}/inputs/${path}"
done
copy_file artifacts/experiences/swan_enumgrpo.json "${setup}/inputs/experience_pool.json"
copy_tree exp/eval/pure_sql_spider "${setup}/results"

# 7. Gold-oracle routing diagnostic.
setup="gold_oracle_routing"
copy_common_runtime "${setup}"
copy_tree experiments/rule_router "${setup}/code/experiment"
copy_file swan/evaluation.jsonl "${setup}/inputs/swan_evaluation.jsonl"
copy_tree exp/eval/gold_oracle_pair_swan_v1 "${setup}/results"

find "${STAGING}" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "${STAGING}/MANIFEST.tsv"

tar -C "${ARCHIVES_DIR}" -cf - "${BUNDLE_NAME}" | zstd -T0 -8 -o "${PARTIAL}"
mv "${PARTIAL}" "${ARCHIVE}"

printf 'bundle=%s\narchive=%s\nfiles=%s\nbytes=%s\n' \
  "${STAGING}" "${ARCHIVE}" \
  "$(find "${STAGING}" -type f | wc -l)" \
  "$(stat -c '%s' "${ARCHIVE}")"
