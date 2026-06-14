#!/usr/bin/env bash
#
# Fetch PiDiNet checkpoints into weights/.
#
# Source: github.com/hellozhuo/pidinet (BSDS500-trained, table5 config).
# Files served directly from the repo's `trained_models/` directory; no
# Drive / OAuth required. Idempotent — skips files that already exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/weights"
mkdir -p "${WEIGHTS_DIR}"
cd "${WEIGHTS_DIR}"

BASE_URL="https://raw.githubusercontent.com/hellozhuo/pidinet/master/trained_models"

# (local_filename, upstream_filename) — local names normalized to
# `pidinet_<variant>.pth` to match what `rgb_edge.PiDiNetEdge` expects.
declare -a FILES=(
  "pidinet_full.pth:table5_pidinet.pth"
  "pidinet_small.pth:table5_pidinet-small.pth"
  "pidinet_tiny.pth:table5_pidinet-tiny.pth"
)

fetch_one() {
  local local_name="$1"
  local upstream="$2"
  if [[ -f "${local_name}" ]]; then
    echo "✓ ${local_name} already present, skipping."
    return
  fi
  echo "↓ Fetching ${local_name}  (upstream: ${upstream}) ..."
  curl -fSL "${BASE_URL}/${upstream}" -o "${local_name}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${local_name}"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${local_name}"
  fi
}

for entry in "${FILES[@]}"; do
  local_name="${entry%%:*}"
  upstream="${entry##*:}"
  fetch_one "${local_name}" "${upstream}"
done

echo
echo "Done. Checkpoints in ${WEIGHTS_DIR}/"
ls -lh "${WEIGHTS_DIR}"/pidinet_*.pth
