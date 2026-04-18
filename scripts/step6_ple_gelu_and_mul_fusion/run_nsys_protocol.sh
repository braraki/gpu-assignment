#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-baseline}"
VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/vllm}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol}"
NSYS_DIR="${RESULTS_ROOT}/nsys"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
WARMUP_SECONDS="${WARMUP_SECONDS:-200}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-30}"

case "$MODE" in
  baseline|ple-gelu-and-mul-fusion) ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$NSYS_DIR"
cd "$VLLM_DIR"
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

echo "Shell 1: this script launches the server under nsys."
echo "Shell 2: after the server is ready, drive traffic with:"
echo "  scripts/step6_ple_gelu_and_mul_fusion/run_nsys_load.sh ${MODE}"
echo
echo "Server URL: http://localhost:${PORT}"
echo "Trace output prefix: ${NSYS_DIR}/${MODE//-/_}_compiled"
echo "Warm-up before capture: ${WARMUP_SECONDS}s"
echo "Capture duration: ${CAPTURE_SECONDS}s"
echo
echo "If you changed C++/CUDA sources, rebuild first with:"
echo "  cd ~/vllm && source .venv/bin/activate && uv pip install -e . --torch-backend=auto"

export VLLM_NVTX_SCOPES_FOR_PROFILING="${VLLM_NVTX_SCOPES_FOR_PROFILING:-1}"
export VLLM_CUSTOM_SCOPES_FOR_PROFILING="${VLLM_CUSTOM_SCOPES_FOR_PROFILING:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --delay "$WARMUP_SECONDS" \
  --duration "$CAPTURE_SECONDS" \
  --output "${NSYS_DIR}/${MODE//-/_}_compiled" \
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
