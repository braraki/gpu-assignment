#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROVIDER="${PROVIDER:-baseline_compiled}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
HEAD_DIMS="${HEAD_DIMS:-}"
NUM_HEADS="${NUM_HEADS:-}"
NUM_KV_HEADS="${NUM_KV_HEADS:-}"
EPS="${EPS:-}"
RUN_NAME="${RUN_NAME:-${PROVIDER}_tokens$(echo "${NUM_TOKENS}" | tr ' ' '-')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/results/part3-qk-norm-rope-fusion/microbench/compiler_dumps}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
BENCHMARK_OUTPUT_DIR="${RUN_DIR}/benchmark_outputs"
LOG_PATH="${RUN_DIR}/torch_output_code.log"

case "$PROVIDER" in
  baseline_compiled|qk_fusion_compiled|attention_prep_compiled)
    ;;
  *)
    echo "PROVIDER must be baseline_compiled, qk_fusion_compiled, or attention_prep_compiled, got: ${PROVIDER}" >&2
    exit 1
    ;;
esac

activate_vllm_venv
if [[ "$PROVIDER" == "attention_prep_compiled" ]]; then
  require_fused_qkv_norm_rope_vnorm_op
fi

mkdir -p "$BENCHMARK_OUTPUT_DIR"

export VLLM_CACHE_ROOT="${RUN_DIR}/vllm_cache"
export VLLM_COMPILE_CACHE_SAVE_FORMAT="${VLLM_COMPILE_CACHE_SAVE_FORMAT:-unpacked}"
export TORCH_LOGS="${TORCH_LOGS:-output_code,recompiles,graph_breaks}"

args=(
  benchmarks/kernels/benchmark_qkv_norm_rope_vnorm.py
  --model "$MODEL"
  --dtype "$DTYPE"
  --providers "$PROVIDER"
  --output-dir "$BENCHMARK_OUTPUT_DIR"
  --skip-correctness-check
  --num-tokens
)

for token_count in $NUM_TOKENS; do
  args+=("$token_count")
done

if [[ -n "$HEAD_DIMS" ]]; then
  args+=(--head-dim)
  for head_dim in $HEAD_DIMS; do
    args+=("$head_dim")
  done
fi

if [[ -n "$NUM_HEADS" ]]; then
  args+=(--num-heads "$NUM_HEADS")
fi

if [[ -n "$NUM_KV_HEADS" ]]; then
  args+=(--num-kv-heads "$NUM_KV_HEADS")
fi

if [[ -n "$EPS" ]]; then
  args+=(--eps "$EPS")
fi

echo "Dumping compiler output for ${PROVIDER}"
echo "Run directory: ${RUN_DIR}"
echo "Torch logs: ${TORCH_LOGS}"
echo "VLLM cache root: ${VLLM_CACHE_ROOT}"

python "${args[@]}" 2>&1 | tee "${LOG_PATH}"

echo
echo "Generated:"
echo "  ${LOG_PATH}"
echo "  ${BENCHMARK_OUTPUT_DIR}"
echo "  ${VLLM_CACHE_ROOT}/torch_compile_cache"
