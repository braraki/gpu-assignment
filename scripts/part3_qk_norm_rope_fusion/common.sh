#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/vllm}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_ROOT}/results/part3-qk-norm-rope-fusion}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}}"

function activate_vllm_venv() {
  cd "$VLLM_DIR"
  source .venv/bin/activate
}

function require_fused_qkv_norm_rope_vnorm_op() {
  if python - <<'PY' >/dev/null 2>&1
import vllm._custom_ops  # noqa: F401
import torch
raise SystemExit(
    0
    if hasattr(torch.ops, "vllm")
    and hasattr(torch.ops.vllm, "fused_qkv_norm_rope_vnorm")
    else 1
)
PY
  then
    return 0
  fi

  echo "Missing torch.ops.vllm.fused_qkv_norm_rope_vnorm in the Python Triton registration path." >&2
  echo "Check that Triton is available and that the local vLLM Python sources are being imported." >&2
  return 1
}

function export_profiling_scopes() {
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export VLLM_NVTX_SCOPES_FOR_PROFILING="${VLLM_NVTX_SCOPES_FOR_PROFILING:-1}"
  export VLLM_CUSTOM_SCOPES_FOR_PROFILING="${VLLM_CUSTOM_SCOPES_FOR_PROFILING:-0}"
}

function launch_vllm_server() {
  local kernel_experiment="$1"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
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
    --gemma4-kernel-experiment "$kernel_experiment"
}
