#!/usr/bin/env bash
# run_heldout.sh -- Held-out DB (heldout) training + evaluation
#
# For each of the 4 SWAN databases, trains on the other 3 DBs' questions
# from learning.jsonl and evaluates on the held-out DB's questions from
# evaluation.jsonl, repeated K times.
#
# Usage:
#   bash run_heldout.sh [OPTIONS]
#
# Options:
#   -c CONFIG       Path to training config YAML  (default: learning/config.yaml)
#   --train TRAIN   Source train JSONL             (default: swan/learning.jsonl)
#   --test  TEST    Source test  JSONL             (default: swan/evaluation.jsonl)
#   -o OUTPUT_DIR   Root dir for eval outputs      (default: exp/heldout_eval_<timestamp>)
#   -k K            Eval repetitions per fold      (default: 3)
#   --skip-runs     Skip agent runs; re-aggregate existing eval_summary.json files
#
# Any flags not listed above are forwarded to learning.cli (training section only).
#
# To skip completed sections, comment out the relevant blocks below.
# Training resumes automatically from checkpoint.json if a fold was interrupted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CONFIG="learning/config.yaml"
TRAIN_PATH="swan/learning.jsonl"
TEST_PATH="swan/evaluation.jsonl"
OUTPUT_DIR="exp/heldout_eval_${TIMESTAMP}"
K=3
SKIP_RUNS=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)    CONFIG="$2";     shift 2 ;;
        --train)        TRAIN_PATH="$2"; shift 2 ;;
        --test)         TEST_PATH="$2";  shift 2 ;;
        -o|--output)    OUTPUT_DIR="$2"; shift 2 ;;
        -k)             K="$2";          shift 2 ;;
        --skip-runs)    SKIP_RUNS=1;     shift   ;;
        -h|--help)
            awk 'NR > 1 && NR <= 22 && /^#/ { sub(/^# ?/, ""); print }' "$0"
            exit 0
            ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Resolve relative paths from repo root
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$TRAIN_PATH" != /* ]] && TRAIN_PATH="${REPO_ROOT}/${TRAIN_PATH}"
[[ "$TEST_PATH"  != /* ]] && TEST_PATH="${REPO_ROOT}/${TEST_PATH}"

SPLITS_DIR="${REPO_ROOT}/exp/learning/heldout_splits"
EXP_BASE="${REPO_ROOT}/exp/learning"

PYTHON="${PYTHON:-python3}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run_heldout.log"

log() { echo "[$(date "+%H:%M:%S")] $*" | tee -a "${LOG_FILE}"; }

log "=== run_heldout.sh ==="
log "K=${K}  OUTPUT_DIR=${OUTPUT_DIR}  SKIP_RUNS=${SKIP_RUNS}"

# Resolve the latest experienced_prompt.md for a given fold.
prompt_for() {
    local fold="$1"
    local latest
    latest=$(ls -1d "${EXP_BASE}/${fold}/20"* 2>/dev/null | sort | tail -1)
    echo "${latest}/experienced_prompt.md"
}

# ---------------------------------------------------------------------------
# Step 1: generate filtered JSONL split files (idempotent -- safe to re-run)
# ---------------------------------------------------------------------------
# echo "=== Generating heldout splits ==="
# python - << PYEOF
# from learning.heldout_runner import build_heldout_splits, ALL_DATABASES
# from pathlib import Path
#
# splits = Path("${SPLITS_DIR}")
# train  = Path("${TRAIN_PATH}")
# test   = Path("${TEST_PATH}")
#
# for db in ALL_DATABASES:
#     t, e = build_heldout_splits(train, test, db, splits)
#     print(f"  {db}: train={t.name} ({sum(1 for l in open(t) if l.strip())} q)"
#           f"  eval={e.name} ({sum(1 for l in open(e) if l.strip())} q)")
# PYEOF
# echo ""

# ---------------------------------------------------------------------------
# Step 2: training -- one fresh process per fold
# (comment out completed folds to skip)
# ---------------------------------------------------------------------------

# echo "=== Fold training: california_schools ==="
# python -m learning.cli \
#     --config        "${CONFIG}" \
#     --exp_id        heldout_california_schools \
#     --practice_path "${SPLITS_DIR}/train_california_schools.jsonl" \
#     --eval_path     "${SPLITS_DIR}/eval_california_schools.jsonl" \
#     "${EXTRA_ARGS[@]}"
#
# echo "=== Fold training: european_football_2 ==="
# python -m learning.cli \
#     --config        "${CONFIG}" \
#     --exp_id        heldout_european_football_2 \
#     --practice_path "${SPLITS_DIR}/train_european_football_2.jsonl" \
#     --eval_path     "${SPLITS_DIR}/eval_european_football_2.jsonl" \
#     "${EXTRA_ARGS[@]}"
#
# echo "=== Fold training: formula_1 ==="
# python -m learning.cli \
#     --config        "${CONFIG}" \
#     --exp_id        heldout_formula_1 \
#     --practice_path "${SPLITS_DIR}/train_formula_1.jsonl" \
#     --eval_path     "${SPLITS_DIR}/eval_formula_1.jsonl" \
#     "${EXTRA_ARGS[@]}"
#
# echo "=== Fold training: superhero ==="
# python -m learning.cli \
#     --config        "${CONFIG}" \
#     --exp_id        heldout_superhero \
#     --practice_path "${SPLITS_DIR}/train_superhero.jsonl" \
#     --eval_path     "${SPLITS_DIR}/eval_superhero.jsonl" \
#     "${EXTRA_ARGS[@]}"
#
# echo ""
# echo "=== Training complete ==="
# echo ""

# ---------------------------------------------------------------------------
# Step 3: evaluation -- K repetitions per fold, then aggregate mean ± SD
# Layout: <OUTPUT_DIR>/<db>/run_1/  run_2/  ...
# ---------------------------------------------------------------------------
log "=== heldout evaluation  (K=${K}  output: ${OUTPUT_DIR}) ==="

run_eval_fold() {
    local db="$1"
    local fold="heldout_${db}"
    local eval_split="${SPLITS_DIR}/eval_${db}.jsonl"
    local prompt
    prompt=$(prompt_for "${fold}")

    if [[ ! -f "${prompt}" ]]; then
        log "  ERROR: experienced_prompt.md not found for ${fold}. Run training first." 
        return 1
    fi

    log "--- Eval fold: ${db}  (K=${K}) ---"
    log "  prompt     : ${prompt}"
    log "  eval split : ${eval_split}"

    if [[ "$SKIP_RUNS" -eq 0 ]]; then
        for (( i=1; i<=K; i++ )); do
            local run_dir="${OUTPUT_DIR}/${db}/run_${i}"
            mkdir -p "${run_dir}"
            log "  run ${i}/${K} -> ${run_dir}"

            set +e
            (cd "${REPO_ROOT}" && "${PYTHON}" run_swan_main.py \
                --query_file  "${eval_split}" \
                --prompt_file "${prompt}" \
                --out_dir     "${run_dir}" 2>&1 | tee -a "${LOG_FILE}")
            local run_rc=$?
            set -e
            [[ $run_rc -ne 0 ]] && log "  WARNING: run_swan_main.py exited ${run_rc} (proceeding to eval)"

            set +e
            (cd "${REPO_ROOT}" && "${PYTHON}" eval_swan.py \
                --run_dir    "${run_dir}" \
                --query_file "${eval_split}" \
                --json > "${run_dir}/eval_summary.json" 2>> "${LOG_FILE}")
            local eval_rc=$?
            set -e
            [[ $eval_rc -ne 0 ]] && log "  WARNING: eval_swan.py exited ${eval_rc}"
            log "  run ${i} scored -> ${run_dir}/eval_summary.json"
        done
    else
        log "  --skip-runs: skipping agent runs for ${db}"
    fi
}

run_eval_fold california_schools
run_eval_fold european_football_2
run_eval_fold formula_1
run_eval_fold superhero

# ---------------------------------------------------------------------------
# Step 4: aggregate K runs per fold with mean ± SD
# ---------------------------------------------------------------------------
log "=== Aggregating results ==="

"${PYTHON}" "${REPO_ROOT}/aggregate_runs.py" \
    --base_dir "${OUTPUT_DIR}" \
    --agents   california_schools european_football_2 formula_1 superhero \
    --k        "${K}" \
    2>> "${LOG_FILE}"

log "=== heldout complete ==="
