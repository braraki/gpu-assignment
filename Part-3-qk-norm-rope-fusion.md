# Part 3 Q/K Norm + RoPE Fusion

This document captures the next optimization direction after the decoder
residual fusion pass: **Gemma 4 attention-prep fusion**.

The important change in scope is that Part 3 deliberately moves closer to the
attention path. Part 2 targeted a narrow decoder residual handoff. Part 3
keeps a narrow source-level change, but aims at a more ambitious kernel family:

1. Q RMSNorm
2. K RMSNorm
3. RoPE on Q/K
4. V RMSNorm

Part 3 is structured to mirror Part 2:

- one primary source-level target
- one operator microbenchmark
- one steady-state `nsys` comparison
- one end-to-end `AIPerf` sweep
- one assignment-facing report

## What The Experiment Is

Part 3 uses a phased experiment family.

### Control Experiment

The control experiment is the existing fused Q/K RMSNorm + RoPE path:

```bash
--gemma4-kernel-experiment qk-norm-rope-fusion-512
```

That path is useful for two reasons:

- it already exists in `vLLM`
- it proves that the attention-prep region is a valid source-level optimization
  target for Gemma 4

### New Kernel Deliverable

The new Part 3 kernel is:

```bash
--gemma4-kernel-experiment qkv-norm-rope-vnorm-fusion
```

That mode targets **non-KV-shared Gemma4 attention layers only** and fuses:

1. `q_norm`
2. `k_norm`
3. `rotary_emb(q, k)`
4. `v_norm`

over the same in-memory `qkv` buffer.

The first Part 3 implementation is intentionally narrow:

- CUDA-only
- Gemma4-specific
- only the observed Gemma4 attention signatures
- only non-KV-shared layers

## Where It Occurs In Code

The relevant source block is in `Gemma4Attention.forward`:

- [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py)

The non-shared attention-prep path is conceptually:

```python
q = q.unflatten(-1, (self.num_heads, self.head_dim))
q = self.q_norm(q)
q = q.flatten(-2, -1)

k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
k = self.k_norm(k)
k = k.flatten(-2, -1)
q, k = self.rotary_emb(positions, q, k)

v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
v = self.v_norm(v)
v = v.flatten(-2, -1)
```

Part 3 adds explicit profiling scopes around this attention-prep region so the
affected source block can be found directly in `nsys`.

The existing control op is:

- `torch.ops._C.fused_qk_norm_rope(...)`

The new Part 3 op is:

- `torch.ops.vllm.fused_qkv_norm_rope_vnorm(...)`

The compile-pass work lives in:

- [qk_norm_rope_fusion.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)

The CUDA kernels live in:

- [fused_qknorm_rope_kernel.cu](/Users/brandonaraki/projects/gpu-assignment/vllm/csrc/fused_qknorm_rope_kernel.cu)

## Baseline Evidence From Part 1

Part 3 uses Part 1 as the baseline reference:

- [Part 1 Benchmarking.md](</Users/brandonaraki/projects/gpu-assignment/gpu-assignment/Part 1 Benchmarking.md>)

The Part 1 `nsys` kernel summary shows why Part 3 changes targets instead of
doubling down on another tiny decoder epilogue cleanup.

The top two GEMM families still dominate the run:

- first GEMM: `63.6%`
- second GEMM: `18.7%`
- [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:91)

The steady attention kernels are still visible:

- `kernel_unified_attention_3d`: `2.3%`
- `kernel_unified_attention_2d`: `0.9%`
- [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:93)
- [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:99)

By comparison, the decoder residual handoff targeted in Part 2 was much
smaller:

- `triton_red_fused_add_rms_norm_0`: `0.7%`
- `triton_red_fused_add_rms_norm_2`: `0.4%`
- [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:100)
- [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:104)

This does **not** mean Part 3 is trying to beat the main GEMMs directly.
It means the next ambitious kernel should move closer to the attention-prep
path, not another sub-1% decoder cleanup.

## Why This Is Worth Pursuing

The current Part 1 baseline does **not** expose Q/K norm + RoPE as a clean,
stable named row in `nsys stats`.

That is an attribution problem, not a reason to ignore the path.

Today the saved baseline shows:

- `kernel_unified_attention_3d`
- `kernel_unified_attention_2d`

but not a clean named `fused_qk_norm_rope` row.

So Part 3 has two jobs:

1. prove attribution more directly with compile-pass evidence and explicit NVTX
   scopes
2. test a more ambitious attention-prep kernel than the existing Q/K-only
   control path

The decision logic for Part 3 is:

- if the next kernel is supposed to be more ambitious and more impactful
- and the main run is still dominated by dense compute plus attention-adjacent
  work
- then the next source-level fusion target should be attention prep, not
  another tiny decoder epilogue

That is why Part 3 uses a phased approach:

- `qk-norm-rope-fusion-512` is the proving-ground and control
- `qkv-norm-rope-vnorm-fusion` is the new kernel deliverable

## Operator Microbenchmark

Part 3 includes an operator-level microbenchmark for the exact changed block.

The benchmark lives in:

- [benchmark_qkv_norm_rope_vnorm.py](/Users/brandonaraki/projects/gpu-assignment/vllm/benchmarks/kernels/benchmark_qkv_norm_rope_vnorm.py)

It compares these providers:

- `baseline_eager`
- `baseline_compiled`
- `qk_fusion_compiled`
- `qk_fusion_custom_op`
- `attention_prep_custom_op`

There is also an internal `attention_prep_compiled` provider for compiler-dump
diagnostics.

By default the benchmark:

- uses Gemma4-derived RMSNorm epsilon
- uses `bfloat16`
- benchmarks token counts `1 4 16 64 256 1024`
- benchmarks both Gemma4-relevant head dimensions
- checks correctness on:
  - `q`
  - `k`
  - `v`

### Run It

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/part3_qk_norm_rope_fusion/run_microbenchmark.sh
```

Outputs land under:

- [results/part3-qk-norm-rope-fusion/microbench](/Users/brandonaraki/projects/gpu-assignment/results/part3-qk-norm-rope-fusion/microbench)

The main CSV is:

- `qkv_norm_rope_vnorm_benchmark.csv`

### Dump Compiler-Generated Code

Use the Part 3 dump script:

```bash
cd ~/gpu-assignment
PROVIDER=baseline_compiled \
gpu-assignment/scripts/part3_qk_norm_rope_fusion/dump_microbenchmark_compiled_code.sh
```

or:

```bash
cd ~/gpu-assignment
PROVIDER=attention_prep_compiled \
gpu-assignment/scripts/part3_qk_norm_rope_fusion/dump_microbenchmark_compiled_code.sh
```

This is the main way Part 3 strengthens attribution beyond kernel-name
guessing.

## How To Interpret A Win

The main comparison of interest is:

- `baseline_compiled`
- `attention_prep_custom_op`

The control comparison is:

- `baseline_compiled`
- `qk_fusion_compiled`

Part 3 is only successful if all of the following hold:

- correctness is preserved
- the compiler-dump evidence shows the intended operator shape
- steady-state `nsys` gets cleaner in the new attention-prep scope
- compiled `AIPerf` does not regress relative to the control path
- ideally, compiled `AIPerf` improves relative to both baseline and the Q/K-only
  control

As in Part 2, compiled serving results are authoritative.

## Validation Plan

Use this order of evidence:

1. Part 1 baseline reference
2. Part 3 control sweep: `qk-norm-rope-fusion-512`
3. Part 3 new kernel sweep: `qkv-norm-rope-vnorm-fusion`
4. steady-state `nsys`
5. operator microbenchmark
6. compiler-dump proof

The scripts live in:

- [scripts/part3_qk_norm_rope_fusion](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part3_qk_norm_rope_fusion)

The expected run flow is:

1. baseline or control server launch
2. `AIPerf` sweep
3. matching steady-state `nsys`
4. operator microbenchmark
5. compiler-dump comparison

## Results

Part 3 has been scaffolded, but the experiment artifacts are still pending.

This section should eventually include:

- the Part 3 microbenchmark CSV and plots
- the compiler-dump evidence for baseline vs fused providers
- one baseline or control `nsys` trace and one new-kernel `nsys` trace
- the end-to-end `AIPerf` comparison

### Placeholder Result Table

| Comparison | Current status |
| --- | --- |
| `baseline_compiled` vs `qk_fusion_compiled` | pending |
| `baseline_compiled` vs `attention_prep_custom_op` | pending |
| steady-state `nsys` scope comparison | pending |
| end-to-end `AIPerf` comparison | pending |

### What We Expect To Change

If Part 3 works as intended, we expect:

- cleaner attention-prep NVTX scopes
- fewer separate Q/K/V normalization boundaries in the affected region
- no correctness regression
- no loss of CUDA graph compatibility
- no loss of async scheduling compatibility

The main point of Part 3 is not that Q/K norm + RoPE already dominates the
kernel summary by name.

The point is that the next ambitious kernel target should be chosen by source
position and workload importance, not only by whichever tiny compiler-generated
epilogue row is easiest to name in the current trace.
