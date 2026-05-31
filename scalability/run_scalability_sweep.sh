#!/usr/bin/env bash
# run_scalability_sweep.sh -- Scale SWAN databases and run agent evaluations
# across all scale factors and methods to measure how plan choices, cost,
# and accuracy change as table size grows.
#
# Usage:
#   bash scalability/run_scalability_sweep.sh [OPTIONS]
#
# Options:
#   -o OUTPUT_DIR       Root dir for all outputs     (default: exp/scalability_<timestamp>)
#   -q QUERY_FILE       SWAN query JSONL              (default: swan/evaluation.jsonl)
#   -e EXPERIENCES      Experiences JSON for db_agent_lx
#                                                    (default: exp/learning/haiku/experiences/latest.json)
#   -m METHODS          Comma-separated agent methods to run
#                       Choices: text2sql,blendsql,db_agent,db_agent_lx,db_agent_nolimit
#                                                    (default: all four)
#   -s SCALES           Comma-separated scale factors, e.g. "0.25,0.5,1.0,2.0,4.0"
#                       Overrides per-db defaults in scalability/scale_databases.py
#   --db DB_ID          Only scale/run this database (repeatable)
#   --seed SEED         Determinism seed for scaling (default: 42)
#   --skip-scale        Skip DB scaling; reuse existing scaled DBs under -o/scaled_dbs
#                       (or the directory given by --scaled-dbs-dir)
#   --scaled-dbs-dir D  Use an existing directory of scaled DBs instead of generating
#                       new ones. Implies --skip-scale.
#   --skip-runs         Skip agent runs; only re-run aggregation on existing results
#   --force-scale       Re-create scaled DBs even if they already exist
#   -k K                Repetitions per (scale, method) combination (default: 1)
#   --scale-timeout     Scale per-query timeout linearly with the scale factor
#                       (e.g. at 4x, timeout = 4 * QUERY_TIMEOUT_S). Default: ON.
#                       Prevents larger-scale runs from being unfairly timed out
#                       while still recording whether a run hit the (scaled) limit.
#   --no-scale-timeout  Use a fixed timeout at all scales (matches SemBench's
#                       protocol: if any query exceeds the limit the run is marked
#                       as timed out and reported as such).
#   --max-timeout S     Cap the scaled timeout at S seconds (default: 7200 = 2h).
#                       Only applies when --scale-timeout is active.
#   -h                  Show this help
#
# Required env vars (can live in .env):
#   AGENT_MODEL, LLMOP_MODEL, DB_FILES_DIR, QUERY_TIMEOUT_S, QUERY_CONCURRENCY
#
# Output layout:
#   <OUTPUT_DIR>/
#     scaled_dbs/                       # produced by scalability/scale_databases.py
#       <scale_label>/
#         <db_id>.duckdb
#       scale_manifest.json
#     runs/
#       <scale_label>/
#         <method>/
#           run_<k>/
#             <question_id>.csv
#             eval_summary.json
#             ...
#     scalability_results.json          # aggregated across scales and methods
#     scalability_summary.tsv           # human-readable TSV for plotting
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source .env so bash-level env vars (QUERY_TIMEOUT_S, LLMOP_TIMEOUT_S, …)
# are available for timeout scaling -- Python scripts load it via dotenv,
# but the shell needs it too.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

# Resolve Python from the 'agent' conda environment if not already overridden.
if [[ -z "${PYTHON:-}" ]]; then
    # Try conda run as a lightweight alternative to full activation.
    if command -v conda &>/dev/null; then
        CONDA_PYTHON="$(conda run -n agent which python 2>/dev/null || true)"
        if [[ -n "$CONDA_PYTHON" && -x "$CONDA_PYTHON" ]]; then
            PYTHON="$CONDA_PYTHON"
        fi
    fi
    # Final fallback.
    PYTHON="${PYTHON:-python3}"
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
K=1
OUTPUT_DIR=""
QUERY_FILE="swan/evaluation.jsonl"
EXPERIENCES_FILE="exp/learning/haiku/experiences/latest.json"
METHODS_ARG=""        # empty = all four
SCALES_ARG=""         # empty = use per-db defaults from scalability/scale_databases.py
DB_FILTER_ARGS=()     # --db flags forwarded to scalability/scale_databases.py
SEED=42
SKIP_SCALE=0
SKIP_RUNS=0
FORCE_SCALE=0
SCALED_DBS_DIR_OVERRIDE=""   # set by --scaled-dbs-dir
SCALE_TIMEOUT=1       # 1 = scale timeout linearly with scale factor (default ON)
MAX_TIMEOUT_S=7200    # cap for scaled timeout (2 hours)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--reps)             K="$2"; shift 2 ;;
        -o|--output)           OUTPUT_DIR="$2"; shift 2 ;;
        -q|--query-file)       QUERY_FILE="$2"; shift 2 ;;
        -e|--experiences)      EXPERIENCES_FILE="$2"; shift 2 ;;
        -m|--methods)          METHODS_ARG="$2"; shift 2 ;;
        -s|--scales)           SCALES_ARG="$2"; shift 2 ;;
        --db)                  DB_FILTER_ARGS+=("--db" "$2"); shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --skip-scale)          SKIP_SCALE=1; shift ;;
        --scaled-dbs-dir)      SCALED_DBS_DIR_OVERRIDE="$2"; SKIP_SCALE=1; shift 2 ;;
        --skip-runs)           SKIP_RUNS=1; shift ;;
        --force-scale)         FORCE_SCALE=1; shift ;;
        --scale-timeout)       SCALE_TIMEOUT=1; shift ;;
        --no-scale-timeout)    SCALE_TIMEOUT=0; shift ;;
        --max-timeout)         MAX_TIMEOUT_S="$2"; shift 2 ;;
        -h|--help)
            awk 'NR > 1 && NR <= 51 && /^#/ { sub(/^# ?/, ""); print }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve methods
# ---------------------------------------------------------------------------
ALL_METHODS=(text2sql blendsql db_agent db_agent_lx db_agent_nolimit)
if [[ -n "$METHODS_ARG" ]]; then
    IFS=',' read -ra METHODS <<< "$METHODS_ARG"
else
    METHODS=("${ALL_METHODS[@]}")
fi

declare -A AGENT_SCRIPTS
AGENT_SCRIPTS[text2sql]="run_swan_agentic_text2sql.py"
AGENT_SCRIPTS[blendsql]="run_swan_agentic_blendsql.py"
AGENT_SCRIPTS[db_agent]="run_swan_main.py"
AGENT_SCRIPTS[db_agent_lx]="run_swan_main.py"
AGENT_SCRIPTS[db_agent_nolimit]="run_swan_main.py"

# Extra CLI args per agent (resolved after OUTPUT_DIR is set)
_build_agent_extra() {
    local slug="$1"
    case "$slug" in
        text2sql)         echo "" ;;
        blendsql)         echo "" ;;
        db_agent)         echo "--prompt_file ${REPO_ROOT}/agent/codebase/prompts/agentic_db_plain.md" ;;
        db_agent_lx)      echo "--prompt_file ${EXPERIENCES_FILE}" ;;
        db_agent_nolimit) echo "--prompt_file ${REPO_ROOT}/agent/codebase/prompts/agentic_db_nolimit.md" ;;
    esac
}

# ---------------------------------------------------------------------------
# Timestamps and paths
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/exp/scalability_${TIMESTAMP}"
fi
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$QUERY_FILE"       != /* ]] && QUERY_FILE="${REPO_ROOT}/${QUERY_FILE}"
[[ "$EXPERIENCES_FILE" != /* ]] && EXPERIENCES_FILE="${REPO_ROOT}/${EXPERIENCES_FILE}"

SCALED_DBS_DIR="${OUTPUT_DIR}/scaled_dbs"
# Override with explicit path if --scaled-dbs-dir was given.
if [[ -n "$SCALED_DBS_DIR_OVERRIDE" ]]; then
    [[ "$SCALED_DBS_DIR_OVERRIDE" != /* ]] && SCALED_DBS_DIR_OVERRIDE="${REPO_ROOT}/${SCALED_DBS_DIR_OVERRIDE}"
    SCALED_DBS_DIR="$SCALED_DBS_DIR_OVERRIDE"
fi
RUNS_DIR="${OUTPUT_DIR}/runs"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/sweep.log"

log() { echo "[$(date "+%H:%M:%S")] $*" | tee -a "$LOG_FILE"; }

if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: python not found. Set PYTHON=/path/to/python." >&2; exit 1
fi
log "Python interpreter: $PYTHON  ($(${PYTHON} --version 2>&1))"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ "$SKIP_RUNS" -eq 0 ]]; then
    [[ -f "$QUERY_FILE" ]] || { echo "Error: query file not found: $QUERY_FILE" >&2; exit 1; }
    # db_agent_lx needs experiences -- warn only, don't abort (user may not run that method)
    if printf '%s\n' "${METHODS[@]}" | grep -q "^db_agent_lx$"; then
        [[ -f "$EXPERIENCES_FILE" ]] || {
            echo "Warning: experiences file not found: $EXPERIENCES_FILE" >&2
            echo "  db_agent_lx will likely fail. Pass -e /path/to/experiences.json." >&2
        }
    fi
fi

# ---------------------------------------------------------------------------
# Write config
# ---------------------------------------------------------------------------
log "=== run_scalability_sweep.sh ==="
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "SCALED_DBS_DIR=${SCALED_DBS_DIR}"
log "QUERY_FILE=${QUERY_FILE}"
log "METHODS=${METHODS[*]}"
log "K=${K}  SEED=${SEED}"
[[ -n "$SCALES_ARG" ]] && log "SCALES=${SCALES_ARG}"
if [[ "$SCALE_TIMEOUT" -eq 1 ]]; then
    log "Timeout mode: scaled linearly with scale factor (cap=${MAX_TIMEOUT_S}s)"
else
    log "Timeout mode: fixed (QUERY_TIMEOUT_S from env)"
fi

{
    echo "timestamp:        ${TIMESTAMP}"
    echo "query_file:       ${QUERY_FILE}"
    echo "experiences_file: ${EXPERIENCES_FILE}"
    echo "methods:          ${METHODS[*]}"
    echo "K:                ${K}"
    echo "seed:             ${SEED}"
    echo "scales_override:  ${SCALES_ARG:-<per-db defaults>}"
    echo "scale_timeout:    ${SCALE_TIMEOUT}  (max=${MAX_TIMEOUT_S}s)"
} > "${OUTPUT_DIR}/config.txt"

# ---------------------------------------------------------------------------
# Step 1: Scale databases
# ---------------------------------------------------------------------------
if [[ "$SKIP_SCALE" -eq 0 ]]; then
    log "=== Scaling databases ==="

    SCALE_CMD=( "$PYTHON" "${REPO_ROOT}/scalability/scale_databases.py"
                "--out_root"    "$SCALED_DBS_DIR"
                "--query_file"  "$QUERY_FILE"
                "--seed"        "$SEED" )

    [[ -n "$SCALES_ARG" ]] && SCALE_CMD+=("--scales" "$SCALES_ARG")
    [[ "$FORCE_SCALE" -eq 1 ]] && SCALE_CMD+=("--force")
    SCALE_CMD+=("${DB_FILTER_ARGS[@]}")

    log "  cmd: ${SCALE_CMD[*]}"
    (cd "$REPO_ROOT" && "${SCALE_CMD[@]}" 2>&1 | tee -a "$LOG_FILE")
    log "  Scaling complete."
else
    log "Skipping scaling (--skip-scale). Using existing DBs under ${SCALED_DBS_DIR}."
fi

# Discover scale labels from manifest (or fallback to directory scan)
MANIFEST="${SCALED_DBS_DIR}/scale_manifest.json"
if [[ -f "$MANIFEST" ]]; then
    # Extract unique labels that succeeded, preserving order by scale value
    SCALE_LABELS=( $("$PYTHON" -c "
import json, sys
manifest = json.load(open('${MANIFEST}'))
seen = {}
for e in manifest:
    if e['status'] == 'ok' and e['label'] not in seen:
        seen[e['label']] = e['scale']
for label in sorted(seen, key=lambda l: seen[l]):
    print(label)
") )
else
    # Fallback: scan directories
    SCALE_LABELS=()
    if [[ -d "$SCALED_DBS_DIR" ]]; then
        while IFS= read -r d; do
            SCALE_LABELS+=("$(basename "$d")")
        done < <(find "$SCALED_DBS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
    fi
fi

if [[ ${#SCALE_LABELS[@]} -eq 0 ]]; then
    log "Error: no scale labels found under ${SCALED_DBS_DIR}." >&2
    exit 1
fi
log "Scale labels to evaluate: ${SCALE_LABELS[*]}"

# ---------------------------------------------------------------------------
# Step 2: Run agents for each (scale, method, repetition)
# ---------------------------------------------------------------------------
if [[ "$SKIP_RUNS" -eq 0 ]]; then
    log "=== Running agents ==="

    for LABEL in "${SCALE_LABELS[@]}"; do
        DB_DIR="${SCALED_DBS_DIR}/${LABEL}"
        if [[ ! -d "$DB_DIR" ]]; then
            log "  [skip] No DB directory for scale ${LABEL}: ${DB_DIR}"
            continue
        fi

        # Compute timeout for this scale.
        # scale_factor = label "0_50x" -> 0.5, "4_00x" -> 4.0
        SCALE_FACTOR=$("$PYTHON" -c "
label = '${LABEL}'
try:
    print(float(label.rstrip('x').replace('_', '.')))
except Exception:
    print(1.0)
")
        if [[ "$SCALE_TIMEOUT" -eq 1 ]]; then
            BASE_TIMEOUT="${QUERY_TIMEOUT_S:-0}"
            if [[ "$BASE_TIMEOUT" -le 0 ]]; then
                log "  WARNING: QUERY_TIMEOUT_S not set; cannot scale timeout. Using run_swan_main.py default."
                TIMEOUT_ARG=""
                LLMOP_TIMEOUT_ENV=""
            else
                SCALED_TIMEOUT=$("$PYTHON" -c "
import math
base = ${BASE_TIMEOUT}
factor = ${SCALE_FACTOR}
# Only scale up; keep base timeout for sub-1x scales (don't penalise small scales)
effective = base if factor <= 1.0 else int(math.ceil(base * factor))
print(min(effective, ${MAX_TIMEOUT_S}))
")
                log "  QUERY_TIMEOUT_S: base=${BASE_TIMEOUT}s -> ${SCALED_TIMEOUT}s (scale=${SCALE_FACTOR}, cap=${MAX_TIMEOUT_S}s)"
                TIMEOUT_ARG="--timeout_s ${SCALED_TIMEOUT}"

                # Scale LLMOP_TIMEOUT_S for the same reason: llm_reduce and the
                # BlendSQL LLMMap/LLMQA ingredient both go through utils/llm.py
                # (LLMOpHostedModel -> llmop_call_async), so the same env var governs
                # all three. Only scale up; keep base for sub-1x.
                BASE_LLMOP="${LLMOP_TIMEOUT_S:-300}"
                SCALED_LLMOP=$("$PYTHON" -c "
import math
base = ${BASE_LLMOP}
factor = ${SCALE_FACTOR}
effective = base if factor <= 1.0 else int(math.ceil(base * factor))
print(min(effective, 3600))  # hard cap at 1h per individual LLM call
")
                log "  LLMOP_TIMEOUT_S:  base=${BASE_LLMOP}s  -> ${SCALED_LLMOP}s"
                LLMOP_TIMEOUT_ENV="LLMOP_TIMEOUT_S=${SCALED_LLMOP}"
            fi
        else
            TIMEOUT_ARG=""
            LLMOP_TIMEOUT_ENV=""
        fi

        for METHOD in "${METHODS[@]}"; do
            SCRIPT="${AGENT_SCRIPTS[$METHOD]}"
            EXTRA=$(_build_agent_extra "$METHOD")

            for (( i=1; i<=K; i++ )); do
                RUN_DIR="${RUNS_DIR}/${LABEL}/${METHOD}/run_${i}"
                mkdir -p "$RUN_DIR"

                log "--- scale=${LABEL}  method=${METHOD}  run=${i}/${K} ---"
                log "  db_dir: ${DB_DIR}"
                log "  output: ${RUN_DIR}"

                CMD=( "$PYTHON" "${REPO_ROOT}/${SCRIPT}"
                      --query_file "$QUERY_FILE"
                      --db_dir     "$DB_DIR"
                      --out_dir    "$RUN_DIR" )

                # If --db filters are active, restrict the agent run to those
                # databases too so it doesn't error on missing scaled DB files.
                for DB_FILT_ARG in "${DB_FILTER_ARGS[@]}"; do
                    if [[ "$DB_FILT_ARG" != "--db" ]]; then
                        CMD+=( "--run_db" "$DB_FILT_ARG" )
                    fi
                done

                if [[ -n "$EXTRA" ]]; then
                    read -ra EXTRA_ARR <<< "$EXTRA"
                    CMD+=( "${EXTRA_ARR[@]}" )
                fi

                if [[ -n "$TIMEOUT_ARG" ]]; then
                    read -ra TIMEOUT_ARR <<< "$TIMEOUT_ARG"
                    CMD+=( "${TIMEOUT_ARR[@]}" )
                fi

                log "  cmd: ${CMD[*]}"
                [[ -n "$LLMOP_TIMEOUT_ENV" ]] && log "  env: ${LLMOP_TIMEOUT_ENV}"

                set +e
                (cd "$REPO_ROOT" && env ${LLMOP_TIMEOUT_ENV} "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE")
                RUN_RC=$?
                set -e
                [[ $RUN_RC -ne 0 ]] && log "  WARNING: agent run exited with code ${RUN_RC}"

                # Evaluate immediately after each run
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
                [[ $EVAL_RC -ne 0 ]] && log "  WARNING: eval_swan.py exited with code ${EVAL_RC}"
                log "  eval done -> ${EVAL_SUMMARY}"
            done
        done
    done
else
    log "Skipping agent runs (--skip-runs)."
fi

# ---------------------------------------------------------------------------
# Step 3: Aggregate results across scales and methods
# ---------------------------------------------------------------------------
log "=== Aggregating scalability results ==="

"$PYTHON" - <<PYEOF 2>> "$LOG_FILE"
import json, math, sys
from pathlib import Path

runs_dir   = Path("${RUNS_DIR}")
out_dir    = Path("${OUTPUT_DIR}")
scale_labels = "${SCALE_LABELS[*]}".split() if "${SCALE_LABELS[*]}" else []

methods_all = ["text2sql", "blendsql", "db_agent", "db_agent_lx"]
methods_run = "${METHODS[*]}".split()

# -------------------------------------------------------
# Parse scale label -> float
# -------------------------------------------------------
def label_to_float(label: str) -> float:
    """'0_50x' -> 0.5,  '2_00x' -> 2.0"""
    s = label.rstrip("x").replace("_", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")

# -------------------------------------------------------
# Collect
# -------------------------------------------------------
records = []  # one per (label, method, run)

for label in scale_labels:
    scale_val = label_to_float(label)
    for method in methods_run:
        method_dir = runs_dir / label / method
        if not method_dir.is_dir():
            continue
        for run_dir in sorted(method_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "eval_summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text())
            except Exception as e:
                print(f"  Warning: could not parse {summary_path}: {e}", file=sys.stderr)
                continue

            perf = summary.get("performance", {})
            costs = summary.get("costs", {})
            agent_c = costs.get("agent_planning", {})
            llm_c   = costs.get("llm_op", {})

            # Infer the timeout that was used for this run from results.jsonl
            # (run_swan_main.py records elapsed_s per query; we can't recover timeout
            # directly, but we note n_timed_out from missing CSVs + elapsed)
            n_timed_out = summary.get("n_missing_csv", 0)

            records.append({
                "scale_label":  label,
                "scale":        scale_val,
                "method":       method,
                "run":          run_dir.name,
                "n_total":      summary.get("n_total", 0),
                "n_matched":    summary.get("n_matched", 0),
                "n_timed_out":  n_timed_out,
                "success_rate": perf.get("success_rate", None),
                "row_f1":       perf.get("row_level_f1", None),
                "item_f1":      perf.get("item_level_f1", None),
                "avg_elapsed_s":perf.get("avg_elapsed_s", None),
                # Agent planning
                "agent_avg_total_tok":  agent_c.get("avg_total_tokens", None),
                "agent_total_cost_usd": agent_c.get("total_cost_usd", None),
                # LLM-op (llm_map / llm_reduce)
                "llmop_avg_total_tok":  llm_c.get("avg_total_tokens", None),
                "llmop_n_queries":      llm_c.get("n_queries_with_llm_op", None),
                "llmop_avg_wf_len":     llm_c.get("avg_workflow_length", None),
            })

# -------------------------------------------------------
# Aggregate: mean +/- sd across repetitions
# -------------------------------------------------------
def _mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None

def _sd(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

from collections import defaultdict
grouped: dict = defaultdict(list)
for r in records:
    grouped[(r["scale_label"], r["scale"], r["method"])].append(r)

aggregated = []
for (label, scale_val, method), reps in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][2])):
    metrics = ["success_rate", "row_f1", "item_f1", "avg_elapsed_s",
               "agent_avg_total_tok", "agent_total_cost_usd",
               "llmop_avg_total_tok", "llmop_n_queries", "llmop_avg_wf_len",
               "n_timed_out"]
    row: dict = {"scale_label": label, "scale": scale_val, "method": method, "n_reps": len(reps)}
    for m in metrics:
        vals = [r[m] for r in reps]
        row[m + "_mean"] = _mean(vals)
        row[m + "_sd"]   = _sd(vals)
    aggregated.append(row)

# -------------------------------------------------------
# Save JSON
# -------------------------------------------------------
results_path = out_dir / "scalability_results.json"
results_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
print(f"Aggregated results -> {results_path}")

# -------------------------------------------------------
# Save TSV
# -------------------------------------------------------
cols = ["scale", "method", "n_reps",
        "success_rate_mean", "success_rate_sd",
        "row_f1_mean",       "row_f1_sd",
        "item_f1_mean",      "item_f1_sd",
        "avg_elapsed_s_mean",
        "n_timed_out_mean",
        "agent_avg_total_tok_mean", "agent_total_cost_usd_mean",
        "llmop_avg_total_tok_mean", "llmop_avg_wf_len_mean"]

tsv_path = out_dir / "scalability_summary.tsv"
with tsv_path.open("w") as fh:
    fh.write("\t".join(cols) + "\n")
    for row in aggregated:
        def fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        fh.write("\t".join(fmt(row.get(c)) for c in cols) + "\n")
print(f"TSV summary        -> {tsv_path}")

# -------------------------------------------------------
# Print console table
# -------------------------------------------------------
print()
print(f"{'Scale':<10} {'Method':<14} {'Reps':>4}  {'SR%':>6}  {'RowF1%':>7}  {'Elapsed':>8}  {'Timeout':>7}  {'AgentTok':>9}  {'LLMOpTok':>9}")
print("-" * 95)
for row in aggregated:
    sr    = row.get("success_rate_mean")
    rf1   = row.get("row_f1_mean")
    el    = row.get("avg_elapsed_s_mean")
    atk   = row.get("agent_avg_total_tok_mean")
    ltk   = row.get("llmop_avg_total_tok_mean")
    tout  = row.get("n_timed_out_mean")
    fmt_f = lambda v: f"{v:7.2f}" if v is not None else "      -"
    fmt_i = lambda v: f"{int(v):9,}" if v is not None else "        -"
    fmt_e = lambda v: f"{v:8.1f}" if v is not None else "       -"
    fmt_t = lambda v: f"{v:7.1f}" if v is not None else "      -"
    print(
        f"{row['scale_label']:<10} {row['method']:<14} {row['n_reps']:>4}"
        f"  {fmt_f(sr)}  {fmt_f(rf1)}  {fmt_e(el)}  {fmt_t(tout)}  {fmt_i(atk)}  {fmt_i(ltk)}"
    )
PYEOF

log "=== Done ==="
log "Results:  ${OUTPUT_DIR}/scalability_results.json"
log "TSV:      ${OUTPUT_DIR}/scalability_summary.tsv"
log "All logs: ${LOG_FILE}"
