#!/usr/bin/env bash
# run_multi_eval.sh -- Run 4 SWAN agents for K repetitions, evaluate each run,
# and print an aggregated comparison table (mean +/- SD) matching results_comparison.md.
#
# Usage:
#   bash run_multi_eval.sh [OPTIONS]
#
# Options:
#   -k K              Number of repetitions            (default: 3)
#   -o OUTPUT_DIR     Root dir for all outputs         (default: exp/multi_eval_<timestamp>)
#   -q QUERY_FILE     SWAN query JSONL                 (default: swan/evaluation.jsonl)
#   -e EXPERIENCES    Experiences JSON for db_agent_lx (default: exp/learning/haiku/experiences/latest.json)
#   --skip-runs       Skip agent runs; only aggregate existing eval_summary.json files
#   -h                Show this help
#
# Agents:
#   text2sql    -- agentic Text2SQL baseline
#   blendsql    -- agentic BlendSQL baseline
#   db_agent    -- DB Agent (no experiences)
#   db_agent_lx -- DB Agent with learned experiences
#                 Override label with LEARNED_AGENT_SLUG=...
#
# Required env vars (can live in .env):
#   AGENT_MODEL, LLMOP_MODEL, DB_FILES_DIR, QUERY_TIMEOUT_S, QUERY_CONCURRENCY
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
K=3
OUTPUT_DIR=""
QUERY_FILE="swan/evaluation.jsonl"
EXPERIENCES_FILE="exp/learning/haiku/experiences/latest.json"
SKIP_RUNS=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k) K="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -q) QUERY_FILE="$2"; shift 2 ;;
        -e) EXPERIENCES_FILE="$2"; shift 2 ;;
        --skip-runs) SKIP_RUNS=1; shift ;;
        -h|--help)
            awk 'NR > 1 && NR <= 25 && /^#/ { sub(/^# ?/, ""); print }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: -k must be a positive integer, got: $K" >&2; exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/exp/multi_eval_${TIMESTAMP}"
fi

# Resolve relative paths from repo root
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$QUERY_FILE"       != /* ]] && QUERY_FILE="${REPO_ROOT}/${QUERY_FILE}"
[[ "$EXPERIENCES_FILE" != /* ]] && EXPERIENCES_FILE="${REPO_ROOT}/${EXPERIENCES_FILE}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ "$SKIP_RUNS" -eq 0 ]]; then
    if [[ ! -f "$QUERY_FILE" ]]; then
        echo "Error: query file not found: $QUERY_FILE" >&2; exit 1
    fi
    if [[ ! -f "$EXPERIENCES_FILE" ]]; then
        echo "Error: experiences file not found: $EXPERIENCES_FILE" >&2
        echo "  Pass -e /path/to/experiences.json to specify a different path." >&2
        exit 1
    fi
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: python not found. Set PYTHON=/path/to/python to override." >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------
LEARNED_AGENT_SLUG="${LEARNED_AGENT_SLUG:-db_agent_lx}"
AGENT_SLUGS=("$LEARNED_AGENT_SLUG")

declare -A AGENT_SCRIPTS
AGENT_SCRIPTS[text2sql]="run_swan_agentic_text2sql.py"
AGENT_SCRIPTS[blendsql]="run_swan_agentic_blendsql.py"
AGENT_SCRIPTS[db_agent]="run_swan_main.py"
AGENT_SCRIPTS[db_agent_lx]="run_swan_main.py"
AGENT_SCRIPTS["$LEARNED_AGENT_SLUG"]="run_swan_main.py"

# Extra CLI args per agent (will be word-split via read -ra)
declare -A AGENT_EXTRA
AGENT_EXTRA[text2sql]=""
AGENT_EXTRA[blendsql]=""
AGENT_EXTRA[db_agent]="--prompt_file ${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md"
AGENT_EXTRA[db_agent_lx]="--prompt_file ${EXPERIENCES_FILE}"
AGENT_EXTRA["$LEARNED_AGENT_SLUG"]="--prompt_file ${EXPERIENCES_FILE}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/run_multi_eval.log"

log() { echo "[$(date "+%H:%M:%S")] $*" | tee -a "$LOG_FILE"; }

log "=== run_multi_eval.sh ==="
log "K=${K}  OUTPUT_DIR=${OUTPUT_DIR}"
log "QUERY_FILE=${QUERY_FILE}"
log "EXPERIENCES_FILE=${EXPERIENCES_FILE}"
log "SKIP_RUNS=${SKIP_RUNS}"

{
    echo "timestamp:        ${TIMESTAMP}"
    echo "K:                ${K}"
    echo "query_file:       ${QUERY_FILE}"
    echo "experiences_file: ${EXPERIENCES_FILE}"
    echo "agents:           ${AGENT_SLUGS[*]}"
} > "${OUTPUT_DIR}/config.txt"

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------
if [[ "$SKIP_RUNS" -eq 0 ]]; then
    for SLUG in "${AGENT_SLUGS[@]}"; do
        SCRIPT="${AGENT_SCRIPTS[$SLUG]}"
        EXTRA="${AGENT_EXTRA[$SLUG]}"

        for (( i=1; i<=K; i++ )); do
            RUN_DIR="${OUTPUT_DIR}/${SLUG}/run_${i}"
            mkdir -p "$RUN_DIR"

            log "--- Agent: ${SLUG}  Run: ${i}/${K} ---"
            log "  output: ${RUN_DIR}"

            CMD=( "$PYTHON" "${REPO_ROOT}/${SCRIPT}"
                  --query_file "$QUERY_FILE"
                  --out_dir    "$RUN_DIR" )

            if [[ -n "$EXTRA" ]]; then
                read -ra EXTRA_ARR <<< "$EXTRA"
                CMD+=( "${EXTRA_ARR[@]}" )
            fi

            log "  cmd: ${CMD[*]}"

            set +e
            (cd "$REPO_ROOT" && "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE")
            RUN_RC=$?
            set -e

            if [[ $RUN_RC -ne 0 ]]; then
                log "  WARNING: agent run exited with code ${RUN_RC} (proceeding to eval)"
            fi

            # Evaluate this run
            log "  evaluating ${RUN_DIR}..."
            EVAL_SUMMARY="${RUN_DIR}/eval_summary.json"

            set +e
            (cd "$REPO_ROOT" && \
                "$PYTHON" eval_swan.py \
                    --run_dir    "$RUN_DIR" \
                    --query_file "$QUERY_FILE" \
                    --json > "$EVAL_SUMMARY" 2>> "$LOG_FILE"
            )
            EVAL_RC=$?
            set -e

            if [[ $EVAL_RC -ne 0 ]]; then
                log "  WARNING: eval_swan.py exited with code ${EVAL_RC}"
            else
                log "  eval done -> ${EVAL_SUMMARY}"
            fi
        done
    done
else
    log "Skipping agent runs (--skip-runs). Searching for existing eval_summary.json files..."
fi

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
log "=== Aggregating results ==="

"$PYTHON" "${REPO_ROOT}/aggregate_runs.py" \
    --base_dir "$OUTPUT_DIR" \
    --agents   "${AGENT_SLUGS[@]}" \
    --k        "$K" \
    2>> "$LOG_FILE"

log "=== Done ==="
