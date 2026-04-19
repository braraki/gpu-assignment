#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

BASE_URL="${BASE_URL:-http://localhost:8000}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_ROOT}/results/part4-qkv-norm-rope-vnorm-kvcache-fusion}"
RUN_SET="${RUN_SET:-qkv-norm-rope-vnorm-kvcache-fusion}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
REQUEST_COUNT="${REQUEST_COUNT:-128}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-8}"

if (( $# > 0 )); then
  CONCURRENCIES=("$@")
else
  CONCURRENCIES=(1 2 4 8)
fi

source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"
mkdir -p "${RESULTS_ROOT}/${RUN_SET}"

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

echo "Running AIPerf sweep for ${RUN_SET}"
echo "Base URL: ${BASE_URL}"
echo "Concurrencies: ${CONCURRENCIES[*]}"

for c in "${CONCURRENCIES[@]}"; do
  echo "=== ${RUN_SET} concurrency ${c} ==="
  aiperf profile \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --url "$BASE_URL" \
    --endpoint-type chat \
    --endpoint v1/chat/completions \
    --streaming \
    --ui simple \
    --concurrency "$c" \
    --request-count "$REQUEST_COUNT" \
    --warmup-request-count "$WARMUP_REQUEST_COUNT" \
    --synthetic-input-tokens-mean 512 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --random-seed 0 \
    --artifact-dir "${RESULTS_ROOT}/${RUN_SET}/c${c}"
done
