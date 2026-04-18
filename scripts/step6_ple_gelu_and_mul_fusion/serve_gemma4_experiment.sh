#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-baseline}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/vllm}"

case "$MODE" in
  baseline|ple-gelu-and-mul-fusion) ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Expected one of: baseline, ple-gelu-and-mul-fusion" >&2
    exit 1
    ;;
esac

cd "$VLLM_DIR"
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache
echo "Compile cache cleared for mode: $MODE"
echo "If you changed C++/CUDA sources, rebuild first with:"
echo "  uv pip install -e . --torch-backend=auto"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --language-model-only \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --async-scheduling \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --gemma4-kernel-experiment "$MODE"
