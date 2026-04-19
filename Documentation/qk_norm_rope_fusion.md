# Gemma 4 Attention-Prep Fusion Notes

## Goal

Part 3 now focuses on the full non-KV-shared Gemma4 attention-prep block:

- `q_norm`
- `k_norm`
- `rotary_emb(q, k)`
- `v_norm`

The new kernel path fuses that block over packed `qkv` by introducing:

- `qkv-norm-rope-vnorm-fusion`

## Why This Experiment Exists

This remains a stronger next target than another decoder epilogue cleanup.

The Part 1 baseline still shows:

- two dominant GEMM families at `63.6%` and `18.7%`
- visible attention kernels at `2.3%` and `0.9%`
- decoder residual kernels that were far smaller than that

So the practical next step is to move closer to the attention path while
keeping the source-level change narrow enough to validate cleanly.

## Gemma 4 Attention Layout

For the relevant non-KV-shared layers, Gemma4 attention prep is:

1. `q_norm`
2. `k_norm`
3. `rotary_emb(q, k)`
4. `v_norm`

The current Gemma4 attention signatures of interest are:

- `(256, 8, 1)`
- `(512, 8, 1)`

The first Part 3 kernel iteration targets only those signatures.

## Public Experiment Modes

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment qkv-norm-rope-vnorm-fusion
```

Mode semantics:

- `baseline`: existing decoder and attention behavior
- `qkv-norm-rope-vnorm-fusion`: new CUDA-only mode that fuses Q/K RMSNorm,
  RoPE, and V RMSNorm for non-KV-shared Gemma4 attention-prep blocks

The Part 3 fusion path requires:

- `pass_config.enable_qk_norm_rope_fusion = True`
- `+rms_norm`
- `+rotary_embedding`

## Source Mapping

The source block is in:

- [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py)

The compile pass is in:

- [qk_norm_rope_fusion.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)

The Triton kernel is in:

- [qkv_norm_rope_vnorm_triton.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/kernels/qkv_norm_rope_vnorm_triton.py)

The new Part 3 op is:

- `torch.ops.vllm.fused_qkv_norm_rope_vnorm(...)`

## Cache Hygiene

Switching between baseline and new-kernel modes may require a fresh torch
compile cache:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

The Part 3 Triton path is Python-only and does not require rebuilding `_C`.

## Scripts

Part 3 uses:

- [scripts/part3_qk_norm_rope_fusion](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part3_qk_norm_rope_fusion)

Important entry points:

- `serve_qk_norm_rope_fusion_gemma4.sh`
- `run_aiperf_sweep.sh`
- `run_aiperf_c4_load.sh`
- `run_microbenchmark.sh`
- `dump_microbenchmark_compiled_code.sh`
- `profile_qkv_norm_rope_vnorm_gemma4_nsys.sh`
- `process_nsys_report.sh`

## What To Look For In `nsys`

The Part 3 attention-prep scopes are:

- `gemma4.attention.prep:baseline`
- `gemma4.attention.prep:qkv_norm_rope_vnorm_fusion`

This is important because the current baseline `nsys stats` do not expose
Q/K norm + RoPE as a clean named kernel row. Part 3 is designed to make the
source-level region attributable even when the kernel summary remains indirect.

## Microbenchmark

The operator benchmark is:

- [benchmark_qkv_norm_rope_vnorm.py](/Users/brandonaraki/projects/gpu-assignment/vllm/benchmarks/kernels/benchmark_qkv_norm_rope_vnorm.py)

Default providers:

- `baseline_eager`
- `baseline_compiled`
- `attention_prep_custom_op`

There is also an internal:

- `attention_prep_compiled`

for compiler-dump diagnostics.

## Success Criteria

Advance the new kernel only if all of the following hold:

- the new CUDA op is correct for both Gemma4 head dimensions
- compile-pass tests prove the new rewrite only hits non-KV-shared attention
  prep
- steady-state `nsys` shows a cleaner attention-prep region
- compiled `AIPerf` does not regress versus baseline
- model behavior remains coherent

If the microbenchmark wins but compiled end-to-end serving loses, the compiled
serving result wins.
