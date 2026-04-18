#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/part1-benchmarking}"
LOAD_DIR="${RESULTS_ROOT}/load"
RUN_NAME="${RUN_NAME:-vanilla_gemma4_e2b_c${CONCURRENCY}_single_round}"

mkdir -p "$LOAD_DIR"

python3 "${SCRIPT_DIR}/single_round_chat_load.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --output "${LOAD_DIR}/${RUN_NAME}.json"
