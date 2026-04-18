#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

BASE_URL="${BASE_URL:-http://localhost:8000}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_ROOT}/results/part3-qk-norm-rope-fusion/load}"
RUN_NAME="${RUN_NAME:-qkv_norm_rope_vnorm_fusion_c4_load}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
REQUEST_COUNT="${REQUEST_COUNT:-8192}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-16}"

source "${AIPERF_VENV:-$HOME/aiperf-venv}/bin/activate"
mkdir -p "${RESULTS_ROOT}"

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

echo "Running sustained AIPerf c=4 load"
echo "Artifact dir: ${RESULTS_ROOT}/${RUN_NAME}"

aiperf profile \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --url "$BASE_URL" \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 4 \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count "$WARMUP_REQUEST_COUNT" \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir "${RESULTS_ROOT}/${RUN_NAME}"
