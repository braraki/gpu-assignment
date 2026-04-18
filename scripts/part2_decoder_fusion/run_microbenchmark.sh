#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/results/part2-decoder-fusion/microbench}"
NUM_TOKENS="${NUM_TOKENS:-1 4 16 64 256 1024}"
PROVIDERS="${PROVIDERS:-baseline_eager baseline_compiled fusion_custom_op}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
EPS="${EPS:-}"
SKIP_CORRECTNESS_CHECK="${SKIP_CORRECTNESS_CHECK:-0}"

activate_vllm_venv

mkdir -p "$OUTPUT_DIR"

args=(
  benchmarks/kernels/benchmark_decoder_residual_fusion.py
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

if [[ -n "$HIDDEN_SIZES" ]]; then
  args+=(--hidden-size)
  for hidden_size in $HIDDEN_SIZES; do
    args+=("$hidden_size")
  done
fi

if [[ -n "$EPS" ]]; then
  args+=(--eps "$EPS")
fi

if [[ "$SKIP_CORRECTNESS_CHECK" == "1" ]]; then
  args+=(--skip-correctness-check)
fi

echo "Running decoder residual fusion microbenchmark"
echo "Output directory: $OUTPUT_DIR"
python "${args[@]}"
