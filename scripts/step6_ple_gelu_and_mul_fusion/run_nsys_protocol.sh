#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-baseline}"
VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/vllm}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol}"
NSYS_DIR="${RESULTS_ROOT}/nsys"
LOAD_DIR="${RESULTS_ROOT}/load"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
WARMUP_SECONDS="${WARMUP_SECONDS:-200}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-30}"

case "$MODE" in
  baseline) OUTPUT_PREFIX="baseline_compiled" ;;
  ple-gelu-and-mul-fusion) OUTPUT_PREFIX="ple_gelu_and_mul_fusion_compiled" ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$NSYS_DIR" "$LOAD_DIR"
cd "$VLLM_DIR"
source .venv/bin/activate

echo "Shell 1: start the server and leave it running:"
echo "  gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ${MODE}"
echo
echo "Shell 2: run this script after the server is healthy."
echo
echo "This script self-warms the workload for ${WARMUP_SECONDS}s and only records the next ${CAPTURE_SECONDS}s."
echo "You do not need a separate manual warm-load shell unless you want to pre-warm before starting nsys."

PYTHONPATH="$VLLM_DIR" nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --delay "$WARMUP_SECONDS" \
  --duration "$CAPTURE_SECONDS" \
  --output "${NSYS_DIR}/${OUTPUT_PREFIX}" \
  python -m vllm.benchmarks.serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url "http://localhost:${PORT}" \
    --model "$MODEL" \
    --dataset-name synthetic \
    --num-prompts 128 \
    --random-input-len 512 \
    --random-output-len 128 \
    --ignore-eos \
    --save-result \
    --result-dir "$LOAD_DIR"
