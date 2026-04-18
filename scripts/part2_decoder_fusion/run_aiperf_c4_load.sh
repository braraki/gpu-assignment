#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/part2-decoder-fusion}"
RUN_NAME="${RUN_NAME:-decoder_residual_fusion_c4_load}"
LOAD_DIR="${RESULTS_ROOT}/nsys"
CONCURRENCY="${CONCURRENCY:-4}"
REQUEST_COUNT="${REQUEST_COUNT:-8192}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-16}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"

source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"
mkdir -p "$LOAD_DIR"

MODELS_URL="${BASE_URL%/}/v1/models"
READY_DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))

echo "Waiting for server readiness at ${MODELS_URL}"
until curl --silent --show-error --fail "$MODELS_URL" >/dev/null; do
  if (( SECONDS >= READY_DEADLINE )); then
    echo "Timed out waiting for server readiness after ${READY_TIMEOUT_SECONDS}s" >&2
    exit 1
  fi
  sleep 1
done

echo "Driving sustained AIPerf load for ${RUN_NAME}"
echo "Base URL: ${BASE_URL}"
echo "Concurrency: ${CONCURRENCY}"
echo "Request count: ${REQUEST_COUNT}"

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
  --artifact-dir "${LOAD_DIR}/${RUN_NAME}"
