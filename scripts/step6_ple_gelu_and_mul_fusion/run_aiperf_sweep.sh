#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results}"

case "$MODE" in
  baseline)
    ARTIFACT_ROOT="$RESULTS_ROOT/step6-baseline"
    RUN_PREFIX="baseline"
    ;;
  ple-gelu-and-mul-fusion)
    ARTIFACT_ROOT="$RESULTS_ROOT/step6-ple-gelu-and-mul-fusion"
    RUN_PREFIX="ple_gelu_and_mul_fusion"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$ARTIFACT_ROOT"
source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"

for C in 1 2 4 8; do
  echo "=== ${MODE} concurrency ${C} ==="
  aiperf profile \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --url "$BASE_URL" \
    --endpoint-type chat \
    --endpoint v1/chat/completions \
    --streaming \
    --ui simple \
    --concurrency "${C}" \
    --request-count 128 \
    --warmup-request-count 8 \
    --synthetic-input-tokens-mean 512 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --random-seed 0 \
    --artifact-dir "${ARTIFACT_ROOT}/${RUN_PREFIX}_c${C}"
done
