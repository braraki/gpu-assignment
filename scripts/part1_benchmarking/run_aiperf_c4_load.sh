#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/part1-benchmarking}"
LOAD_DIR="${RESULTS_ROOT}/load"
RUN_NAME="${RUN_NAME:-vanilla_gemma4_e2b_c4_aiperf_like}"
CONCURRENCY="${CONCURRENCY:-4}"
REQUEST_COUNT="${REQUEST_COUNT:-8192}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-16}"
INPUT_TOKENS_MEAN="${INPUT_TOKENS_MEAN:-512}"
INPUT_TOKENS_STDDEV="${INPUT_TOKENS_STDDEV:-0}"
OUTPUT_TOKENS_MEAN="${OUTPUT_TOKENS_MEAN:-128}"
OUTPUT_TOKENS_STDDEV="${OUTPUT_TOKENS_STDDEV:-0}"

mkdir -p "$LOAD_DIR"
source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"

echo "Driving AIPerf load against ${BASE_URL}"
echo "Concurrency: ${CONCURRENCY}"
echo "Request count: ${REQUEST_COUNT}"
echo "Warmup request count: ${WARMUP_REQUEST_COUNT}"
echo "Artifact dir: ${LOAD_DIR}/${RUN_NAME}"

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
  --synthetic-input-tokens-mean "$INPUT_TOKENS_MEAN" \
  --synthetic-input-tokens-stddev "$INPUT_TOKENS_STDDEV" \
  --output-tokens-mean "$OUTPUT_TOKENS_MEAN" \
  --output-tokens-stddev "$OUTPUT_TOKENS_STDDEV" \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir "${LOAD_DIR}/${RUN_NAME}"
