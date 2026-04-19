#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NSYS_DIR="${RESULTS_ROOT}/nsys"
KERNEL_EXPERIMENT="${KERNEL_EXPERIMENT:-qkv-norm-rope-vnorm-kvcache-fusion}"
RUN_NAME="${RUN_NAME:-qkv_norm_rope_vnorm_kvcache_fusion_c4_load}"
WARMUP_SECONDS="${WARMUP_SECONDS:-200}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-30}"
TRACE_NAME="${TRACE_NAME:-${KERNEL_EXPERIMENT}}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_SAMPLE="${NSYS_SAMPLE:-process-tree}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-process-tree}"

mkdir -p "$NSYS_DIR"
activate_vllm_venv
export_profiling_scopes

if [[ "${KERNEL_EXPERIMENT}" == "qkv-norm-rope-vnorm-kvcache-fusion" ]]; then
  require_fused_qkv_norm_rope_vnorm_kvcache_op
fi

echo "Shell 1: this script launches the Part 4 server under nsys."
echo "Shell 2: after the server is ready, run:"
echo "  RUN_NAME=${RUN_NAME} scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_aiperf_c4_load.sh"
echo
echo "Kernel experiment: ${KERNEL_EXPERIMENT}"
echo "Server URL: http://localhost:${PORT}"
echo "Trace output prefix: ${NSYS_DIR}/${TRACE_NAME}"
echo "Warm-up before capture: ${WARMUP_SECONDS}s"
echo "Capture duration: ${CAPTURE_SECONDS}s"
echo "Trace domains: ${NSYS_TRACE}"
echo "CPU sampling: ${NSYS_SAMPLE}"
echo "CPU context switches: ${NSYS_CPUCTXSW}"
echo "VLLM_WORKER_MULTIPROC_METHOD: ${VLLM_WORKER_MULTIPROC_METHOD}"
echo "VLLM_NVTX_SCOPES_FOR_PROFILING: ${VLLM_NVTX_SCOPES_FOR_PROFILING}"
echo "VLLM_CUSTOM_SCOPES_FOR_PROFILING: ${VLLM_CUSTOM_SCOPES_FOR_PROFILING}"

rm -rf ~/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD}" \
VLLM_NVTX_SCOPES_FOR_PROFILING="${VLLM_NVTX_SCOPES_FOR_PROFILING}" \
VLLM_CUSTOM_SCOPES_FOR_PROFILING="${VLLM_CUSTOM_SCOPES_FOR_PROFILING}" \
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
    --gemma4-kernel-experiment "$KERNEL_EXPERIMENT"
