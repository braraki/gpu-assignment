#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NSYS_DIR="${RESULTS_ROOT}/nsys"
WARMUP_SECONDS="${WARMUP_SECONDS:-140}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-30}"
TRACE_NAME="${TRACE_NAME:-decoder_residual_fusion}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_SAMPLE="${NSYS_SAMPLE:-process-tree}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-process-tree}"

mkdir -p "$NSYS_DIR"
activate_vllm_venv
export_profiling_scopes

echo "Shell 1: this script launches the decoder-residual-fusion server under nsys."
echo "Shell 2: after the server is ready, run:"
echo "  RUN_NAME=decoder_residual_fusion_c4_load scripts/part2_decoder_fusion/run_aiperf_c4_load.sh"
echo
echo "Server URL: http://localhost:${PORT}"
echo "Trace output prefix: ${NSYS_DIR}/${TRACE_NAME}"
echo "Warm-up before capture: ${WARMUP_SECONDS}s"
echo "Capture duration: ${CAPTURE_SECONDS}s"
echo "Trace domains: ${NSYS_TRACE}"
echo "CPU sampling: ${NSYS_SAMPLE}"
echo "CPU context switches: ${NSYS_CPUCTXSW}"

rm -rf ~/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
nsys profile \
  --trace "$NSYS_TRACE" \
  --sample "$NSYS_SAMPLE" \
  --cpuctxsw="$NSYS_CPUCTXSW" \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay "$WARMUP_SECONDS" \
  --duration "$CAPTURE_SECONDS" \
  --output "${NSYS_DIR}/${TRACE_NAME}" \
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --host "$HOST" \
    --port "$PORT" \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-chunked-prefill \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --gemma4-kernel-experiment decoder-residual-fusion
