#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol}"
NSYS_DIR="${RESULTS_ROOT}/nsys"
LOAD_DIR="${RESULTS_ROOT}/load"
PORT="${PORT:-8000}"

case "$MODE" in
  baseline) OUTPUT_PREFIX="baseline_compiled" ;;
  ple-gelu-and-mul-fusion) OUTPUT_PREFIX="ple_gelu_and_mul_fusion_compiled" ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$NSYS_DIR" "$LOAD_DIR"

echo "Start the server in a separate shell first:"
echo "  gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ${MODE}"
echo
echo "Then run a warm load in another shell so nsys captures steady state only."

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --delay 15 \
  --duration 30 \
  --output "${NSYS_DIR}/${OUTPUT_PREFIX}" \
  python3 -m vllm.benchmarks.serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url "http://localhost:${PORT}" \
    --model google/gemma-4-E2B-it \
    --dataset-name synthetic \
    --num-prompts 128 \
    --random-input-len 512 \
    --random-output-len 128 \
    --ignore-eos \
    --save-result \
    --result-dir "$LOAD_DIR"
