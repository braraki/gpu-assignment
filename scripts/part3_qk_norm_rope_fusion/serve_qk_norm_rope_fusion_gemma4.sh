#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

KERNEL_EXPERIMENT="${1:-${KERNEL_EXPERIMENT:-qkv-norm-rope-vnorm-fusion}}"

activate_vllm_venv
export_profiling_scopes

echo "Starting Part 3 Gemma 4 server"
echo "Kernel experiment: ${KERNEL_EXPERIMENT}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

launch_vllm_server "${KERNEL_EXPERIMENT}"
