#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/../vllm}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/gpu-assignment-results/part1-benchmarking}"
NSYS_DIR="${RESULTS_ROOT}/nsys"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
WARMUP_SECONDS="${WARMUP_SECONDS:-140}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-30}"
TRACE_NAME="${TRACE_NAME:-vanilla_gemma4_e2b_c4_aiperf_like}"

mkdir -p "$NSYS_DIR"
cd "$VLLM_DIR"
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

echo "Shell 1: this script launches the vanilla vLLM server under nsys."
echo "Shell 2: after the server is ready, run:"
echo "  scripts/part1_benchmarking/run_aiperf_c4_load.sh"
echo
echo "Server URL: http://localhost:${PORT}"
echo "Trace output prefix: ${NSYS_DIR}/${TRACE_NAME}"
echo "Warm-up before capture: ${WARMUP_SECONDS}s"
echo "Capture duration: ${CAPTURE_SECONDS}s"
echo
echo "This default assumes server startup takes about 80s and leaves"
echo "roughly 60s of post-startup warm-up before capture begins."
echo
echo "If you changed vLLM code, rebuild first with:"
echo "  cd ${VLLM_DIR} && source .venv/bin/activate && uv pip install -e . --torch-backend=auto"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_NVTX_SCOPES_FOR_PROFILING="${VLLM_NVTX_SCOPES_FOR_PROFILING:-1}"
export VLLM_CUSTOM_SCOPES_FOR_PROFILING="${VLLM_CUSTOM_SCOPES_FOR_PROFILING:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay "$WARMUP_SECONDS" \
  --duration "$CAPTURE_SECONDS" \
  --output "${NSYS_DIR}/${TRACE_NAME}" \
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
