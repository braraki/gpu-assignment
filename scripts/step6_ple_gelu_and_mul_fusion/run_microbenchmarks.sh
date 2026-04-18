#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

VLLM_DIR="${VLLM_DIR:-${WORKSPACE_ROOT}/vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/gpu-assignment-results/step6-ple-gelu-and-mul-fusion/microbenchmarks}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"

cd "$VLLM_DIR"
source .venv/bin/activate

python benchmarks/kernels/benchmark_ple_gelu_and_mul.py \
  --model "$MODEL" \
  --dtype bfloat16 \
  --num-tokens 1 16 128 1024 4096 \
  --output-dir "$OUTPUT_DIR"
