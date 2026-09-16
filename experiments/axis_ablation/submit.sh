#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/home/lynie/projects/EnumGRPO"
OUTPUT_ROOT="${AXIS_ABLATION_OUTPUT_ROOT:-${REPO_ROOT}/exp/eval/experience_axis_ablation_swan_v1}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: refusing to reuse existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"
sbatch \
  --output="${OUTPUT_ROOT}/logs/slurm-%j.log" \
  --export="ALL,AXIS_ABLATION_OUTPUT_ROOT=${OUTPUT_ROOT}" \
  experiments/axis_ablation/eval.sbatch
