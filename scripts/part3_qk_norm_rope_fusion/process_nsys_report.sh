#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NSYS_DIR="${RESULTS_ROOT}/nsys"
TRACE_NAME="${TRACE_NAME:-qkv-norm-rope-vnorm-fusion}"

REPORT_PREFIX="${NSYS_DIR}/${TRACE_NAME}"
REPORT_FILE="${REPORT_PREFIX}.nsys-rep"
STATS_PREFIX="${REPORT_PREFIX}_stats"

if [[ ! -f "$REPORT_FILE" ]]; then
  echo "Missing nsys report: ${REPORT_FILE}" >&2
  exit 1
fi

mkdir -p "$NSYS_DIR"

echo "Processing ${REPORT_FILE}"
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum,osrt_sum \
  --force-overwrite true \
  --format csv,column \
  --output "${STATS_PREFIX}" \
  "$REPORT_FILE"

echo "Generated:"
echo "  ${STATS_PREFIX}.csv"
echo "  ${STATS_PREFIX}.txt"
