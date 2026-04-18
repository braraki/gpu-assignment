#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol}"
LOAD_DIR="${RESULTS_ROOT}/load"
CONCURRENCY="${CONCURRENCY:-4}"
REQUEST_COUNT="${REQUEST_COUNT:-16384}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-16}"

case "$MODE" in
  baseline|ple-gelu-and-mul-fusion) ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$LOAD_DIR"
source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"

echo "Driving steady-state load against ${BASE_URL} with concurrency ${CONCURRENCY}"

aiperf profile \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --url "$BASE_URL" \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count "$WARMUP_REQUEST_COUNT" \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir "${LOAD_DIR}/${MODE//-/_}_c${CONCURRENCY}"
