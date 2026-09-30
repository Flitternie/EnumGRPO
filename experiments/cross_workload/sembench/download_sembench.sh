#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST="${SEMBENCH_ROOT:-${REPO_ROOT}/datasets/sembench/source}"
REVISION="${SEMBENCH_REVISION:-c814e3807e72d4cf876b852b17e77f3cc94575c2}"
URL="https://github.com/SemBench/SemBench.git"

if [[ -e "${DEST}" ]]; then
  if [[ ! -d "${DEST}/.git" ]]; then
    echo "ERROR: ${DEST} exists but is not a Git checkout" >&2
    exit 1
  fi
else
  mkdir -p "$(dirname "${DEST}")"
  git clone --filter=blob:none --no-checkout "${URL}" "${DEST}"
fi

git -C "${DEST}" sparse-checkout init --cone
git -C "${DEST}" sparse-checkout set \
  files/movie/data/sf_2000 \
  files/movie/query/natural_language \
  files/movie/query/gold_sql \
  src/runner \
  src/scenario/movie
git -C "${DEST}" fetch origin "${REVISION}"
git -C "${DEST}" checkout --detach "${REVISION}"

actual="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${actual}" != "${REVISION}" ]]; then
  echo "ERROR: expected ${REVISION}, got ${actual}" >&2
  exit 1
fi

echo "SemBench ${actual} prepared at ${DEST}"
