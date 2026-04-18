#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/part1-benchmarking}"
NSYS_DIR="${RESULTS_ROOT}/nsys"
TRACE_NAME="${TRACE_NAME:-vanilla_gemma4_e2b_c4_aiperf_like}"

TRACE="${NSYS_DIR}/${TRACE_NAME}.nsys-rep"
OUT="${NSYS_DIR}/${TRACE_NAME}_stats"

if [[ ! -f "$TRACE" ]]; then
  echo "Trace not found: ${TRACE}" >&2
  exit 1
fi

echo "Processing ${TRACE}"

nsys stats "$TRACE" > "${OUT}.txt"
nsys stats --format csv "$TRACE" > "${OUT}.csv"
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
  "$TRACE" > "${OUT}_focused.txt"
nsys export --type sqlite --output "${OUT}" "$TRACE"

echo "Generated:"
echo "  ${OUT}.txt"
echo "  ${OUT}.csv"
echo "  ${OUT}_focused.txt"
echo "  ${OUT}.sqlite"
