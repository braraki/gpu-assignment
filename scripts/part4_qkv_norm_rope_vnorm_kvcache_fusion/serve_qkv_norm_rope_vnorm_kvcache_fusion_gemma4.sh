#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

KERNEL_EXPERIMENT="${1:-${KERNEL_EXPERIMENT:-qkv-norm-rope-vnorm-kvcache-fusion}}"

activate_vllm_venv
export_profiling_scopes

if [[ "${KERNEL_EXPERIMENT}" == "qkv-norm-rope-vnorm-kvcache-fusion" ]]; then
  require_fused_qkv_norm_rope_vnorm_kvcache_op
fi

echo "Starting Part 4 Gemma 4 server"
echo "Kernel experiment: ${KERNEL_EXPERIMENT}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Attention backend: ${ATTENTION_BACKEND}"
echo "VLLM_WORKER_MULTIPROC_METHOD: ${VLLM_WORKER_MULTIPROC_METHOD}"
echo "VLLM_NVTX_SCOPES_FOR_PROFILING: ${VLLM_NVTX_SCOPES_FOR_PROFILING}"
echo "VLLM_CUSTOM_SCOPES_FOR_PROFILING: ${VLLM_CUSTOM_SCOPES_FOR_PROFILING}"

launch_vllm_server "${KERNEL_EXPERIMENT}"
