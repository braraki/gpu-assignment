#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/../vllm}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"

cd "$VLLM_DIR"
source .venv/bin/activate

echo "Starting vanilla vLLM server"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --language-model-only \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --async-scheduling \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
