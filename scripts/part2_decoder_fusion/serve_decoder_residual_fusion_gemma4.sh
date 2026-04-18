#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

activate_vllm_venv
export_profiling_scopes

echo "Starting decoder-residual-fusion Gemma 4 server"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

launch_vllm_server decoder-residual-fusion
