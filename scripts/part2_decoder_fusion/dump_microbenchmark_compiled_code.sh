#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROVIDER="${PROVIDER:-baseline_compiled}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
EPS="${EPS:-}"
RUN_NAME="${RUN_NAME:-${PROVIDER}_tokens$(echo "${NUM_TOKENS}" | tr ' ' '-')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/results/part2-decoder-fusion/microbench/compiler_dumps}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
BENCHMARK_OUTPUT_DIR="${RUN_DIR}/benchmark_outputs"
LOG_PATH="${RUN_DIR}/torch_output_code.log"

case "$PROVIDER" in
  baseline_compiled|fusion_compiled)
    ;;
  *)
    echo "PROVIDER must be baseline_compiled or fusion_compiled, got: ${PROVIDER}" >&2
    exit 1
    ;;
esac

activate_vllm_venv

mkdir -p "$BENCHMARK_OUTPUT_DIR"

export VLLM_CACHE_ROOT="${RUN_DIR}/vllm_cache"
export VLLM_COMPILE_CACHE_SAVE_FORMAT="${VLLM_COMPILE_CACHE_SAVE_FORMAT:-unpacked}"
export TORCH_LOGS="${TORCH_LOGS:-output_code,recompiles,graph_breaks}"

args=(
  benchmarks/kernels/benchmark_decoder_residual_fusion.py
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

if [[ -n "$HIDDEN_SIZES" ]]; then
  args+=(--hidden-size)
  for hidden_size in $HIDDEN_SIZES; do
    args+=("$hidden_size")
  done
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
echo
echo "Next:"
echo "  1. Search ${LOG_PATH} for 'output_code' and 'store the'"
echo "  2. Inspect files under ${VLLM_CACHE_ROOT}/torch_compile_cache/*/rank_0_0/inductor_cache/"
echo "  3. Compare baseline_compiled vs fusion_compiled generated code"
