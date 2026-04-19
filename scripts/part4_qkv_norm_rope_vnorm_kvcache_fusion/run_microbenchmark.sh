#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/results/part4-qkv-norm-rope-vnorm-kvcache-fusion/microbench}"
NUM_TOKENS="${NUM_TOKENS:-1 4 16 64 256 1024}"
PROVIDERS="${PROVIDERS:-baseline_compiled post_gemm_kvcache_custom_op}"
HEAD_DIMS="${HEAD_DIMS:-}"
NUM_HEADS="${NUM_HEADS:-}"
NUM_KV_HEADS="${NUM_KV_HEADS:-}"
EPS="${EPS:-}"
SKIP_CORRECTNESS_CHECK="${SKIP_CORRECTNESS_CHECK:-0}"

activate_vllm_venv
mkdir -p "$OUTPUT_DIR"

if [[ "$SKIP_CORRECTNESS_CHECK" != "1" ]] || [[ " $PROVIDERS " == *" post_gemm_kvcache_custom_op "* ]] || [[ " $PROVIDERS " == *" post_gemm_kvcache_compiled "* ]]; then
  require_fused_qkv_norm_rope_vnorm_kvcache_op
fi

args=(
  benchmarks/kernels/benchmark_qkv_norm_rope_vnorm_kvcache.py
  --model "$MODEL"
  --dtype "$DTYPE"
  --output-dir "$OUTPUT_DIR"
  --num-tokens
)

for token_count in $NUM_TOKENS; do
  args+=("$token_count")
done

args+=(--providers)
for provider in $PROVIDERS; do
  args+=("$provider")
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

if [[ "$SKIP_CORRECTNESS_CHECK" == "1" ]]; then
  args+=(--skip-correctness-check)
fi

echo "Running Part 4 full post-GEMM microbenchmark"
echo "Output directory: $OUTPUT_DIR"
python "${args[@]}"
