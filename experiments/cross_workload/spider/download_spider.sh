#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${SPIDER_DATA_ROOT:-${REPO_ROOT}/datasets/spider1}"
RAW_DIR="${DATA_ROOT}/raw"
ARCHIVE="${RAW_DIR}/spider_data.zip"
UPSTREAM_DIR="${RAW_DIR}/upstream"

SPIDER_CODE_COMMIT="b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c"
SPIDER_ARCHIVE_REVISION="4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
SPIDER_ARCHIVE_SHA256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
SPIDER_ARCHIVE_URL="https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/${SPIDER_ARCHIVE_REVISION}/spider_data.zip"

mkdir -p "${RAW_DIR}"

if [[ ! -f "${ARCHIVE}" ]] || ! echo "${SPIDER_ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check --status; then
  partial="${ARCHIVE}.partial"
  rm -f "${partial}"
  curl -L --fail --retry 4 --retry-delay 3 --connect-timeout 30 \
    -o "${partial}" "${SPIDER_ARCHIVE_URL}"
  echo "${SPIDER_ARCHIVE_SHA256}  ${partial}" | sha256sum --check
  mv "${partial}" "${ARCHIVE}"
fi

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --no-checkout https://github.com/taoyds/spider.git "${UPSTREAM_DIR}"
fi
git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${SPIDER_CODE_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout --detach "${SPIDER_CODE_COMMIT}"

echo "Spider archive: ${ARCHIVE}"
echo "Archive SHA256: $(sha256sum "${ARCHIVE}" | awk '{print $1}')"
echo "Official scorer commit: $(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
