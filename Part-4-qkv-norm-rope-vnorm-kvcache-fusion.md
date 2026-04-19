# Part 4 Full Post-GEMM Attention-Prep-and-Cache Fusion

This document captures the next step after Part 3.

Part 3 proved an important point: fusing only Q/K/V preparation was not enough.
In matched Nsight windows, the pre-attention burst still contained
`reshape_and_cache_kernel_flash`, which meant the post-GEMM handoff into
attention was still split across multiple kernels.

Part 4 pushes one step further and targets the full decoder path between the
QKV GEMM and the attention kernel:

1. Q RMSNorm
2. K RMSNorm
3. RoPE on Q/K
4. V RMSNorm
5. unified KV cache update

Like Part 3, Part 4 is organized as:

- one primary source-level target
- one operator microbenchmark
- one steady-state `nsys` comparison
- one end-to-end `AIPerf` sweep
- one assignment-facing report

## What The Experiment Is

The Part 4 kernel is:

```bash
--gemma4-kernel-experiment qkv-norm-rope-vnorm-kvcache-fusion
```

That mode targets **non-KV-shared Gemma4 decoder attention layers** and fuses:

1. `q_norm`
2. `k_norm`
3. `rotary_emb(q, k)`
4. `v_norm`
5. `unified_kv_cache_update(k, v, layer_name)`

over the same in-memory packed `qkv` buffer.

The first Part 4 implementation is intentionally narrow:

- Triton-only
- CUDA-only
- FlashAttention backend only
- `dcp_world_size == 1`
- non-quantized KV cache only
- Gemma4-specific attention signatures only
- only non-KV-shared layers

## Where It Occurs In Code

The Part 4 source-level target spans two adjacent boundaries:

1. Gemma4 attention prep in `Gemma4Attention.forward`
2. the KV-cache update in `Attention.forward`

The conceptual baseline path is:

```python
q, k, v = qkv.split(...)

q = q_norm(q)
k = k_norm(k)
q, k = rotary_emb(positions, q, k)
v = v_norm(v)

q = q.view(-1, num_heads, head_dim)
k = k.view(-1, num_kv_heads, head_dim)
v = v.view(-1, num_kv_heads, head_dim)
kv_cache_dummy_dep = unified_kv_cache_update(k, v, layer_name)
```

Part 4 replaces that sequence with:

- `torch.ops.vllm.fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update(...)`

The main implementation files are:

- [qkv_norm_rope_vnorm_triton.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/kernels/qkv_norm_rope_vnorm_triton.py)
- [qk_norm_rope_fusion.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- [attention.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/layers/attention/attention.py)

## Why This Is The Next Step After Part 3

Part 3 fused only the attention-prep portion of the post-GEMM window.

That still left a visible cache-update boundary in the matched Nsight traces:

- baseline still showed the usual prep burst plus `reshape_and_cache_kernel_flash`
- the Part 3 fused path still did not collapse the entire pre-attention window

So the Part 4 decision logic is straightforward:

1. Part 3 already tested the smaller “prep only” kernel
2. matched traces still showed remaining post-GEMM work
3. the next worthwhile kernel must absorb the KV-cache update as well

In other words, Part 4 is not “make Part 3 a little faster.”
It is:

- replace the full post-GEMM pre-attention handoff with one fused custom op

## Operator Microbenchmark

Part 4 adds a new operator benchmark for the **full post-GEMM window**.

The benchmark lives in:

- [benchmark_qkv_norm_rope_vnorm_kvcache.py](/Users/brandonaraki/projects/gpu-assignment/vllm/benchmarks/kernels/benchmark_qkv_norm_rope_vnorm_kvcache.py)

It compares:

- `baseline_compiled`
- `post_gemm_kvcache_custom_op`

There is also an internal compiled diagnostic provider:

- `post_gemm_kvcache_compiled`

These names mean:

- `baseline_compiled`: the unfused post-GEMM block under `torch.compile`
- `post_gemm_kvcache_custom_op`: the new Part 4 fused post-GEMM kernel
- `post_gemm_kvcache_compiled`: the same Part 4 fused path, but wrapped in
  `torch.compile` for compiler-dump inspection

This benchmark differs from Part 3 in one important way:

- it needs a real KV-cache context and slot mapping, because cache write is
  part of the fused kernel contract

Outputs land under:

- [results/part4-qkv-norm-rope-vnorm-kvcache-fusion/microbench](/Users/brandonaraki/projects/gpu-assignment/results/part4-qkv-norm-rope-vnorm-kvcache-fusion/microbench)

## How To Interpret A Win

The main comparison of interest is:

- `baseline_compiled`
- `post_gemm_kvcache_custom_op`

Part 4 only counts as a real win if:

- correctness is preserved for Q, K, V, and KV cache contents
- the compiler dump proves the fused op is in the graph
- matched `nsys` windows no longer show `reshape_and_cache_kernel_flash`
- the local window from the QKV GEMM to `kernel_unified_attention_*` is not
  slower than baseline
- compiled `AIPerf` does not regress

As in Parts 2 and 3, compiled serving results are authoritative.

## Validation Plan

Use this order of evidence:

1. Part 3 negative-control result
2. Part 4 full post-GEMM microbenchmark
3. compiler-dump proof
4. steady-state `nsys`
5. end-to-end `AIPerf`

The core checks are:

1. Q output matches the unfused baseline
2. K output matches the unfused baseline
3. V output matches the unfused baseline
4. KV cache contents match the unfused `reshape_and_cache_flash` path
5. `reshape_and_cache_kernel_flash` disappears from the matched Part 4 window

The scripts live in:

- [scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion)

The results root is:

- [results/part4-qkv-norm-rope-vnorm-kvcache-fusion](/Users/brandonaraki/projects/gpu-assignment/results/part4-qkv-norm-rope-vnorm-kvcache-fusion)

## Copy-Paste Commands

### Baseline Server

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/serve_qkv_norm_rope_vnorm_kvcache_fusion_gemma4.sh baseline
```

### Part 4 Fused Server

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/serve_qkv_norm_rope_vnorm_kvcache_fusion_gemma4.sh qkv-norm-rope-vnorm-kvcache-fusion
```

### Baseline AIPerf Sweep

```bash
cd ~/gpu-assignment
RUN_SET=baseline \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_aiperf_sweep.sh
```

### Part 4 AIPerf Sweep

```bash
cd ~/gpu-assignment
RUN_SET=qkv-norm-rope-vnorm-kvcache-fusion \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_aiperf_sweep.sh
```

### Baseline `nsys`

```bash
cd ~/gpu-assignment
KERNEL_EXPERIMENT=baseline \
TRACE_NAME=baseline \
RUN_NAME=baseline_c4_load \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/profile_qkv_norm_rope_vnorm_kvcache_fusion_gemma4_nsys.sh
```

### Baseline Sustained Load For `nsys`

Run this in a second shell after the baseline `nsys` server is ready:

```bash
cd ~/gpu-assignment
RUN_NAME=baseline_c4_load \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_aiperf_c4_load.sh
```

### Part 4 `nsys`

```bash
cd ~/gpu-assignment
KERNEL_EXPERIMENT=qkv-norm-rope-vnorm-kvcache-fusion \
TRACE_NAME=qkv-norm-rope-vnorm-kvcache-fusion \
RUN_NAME=qkv_norm_rope_vnorm_kvcache_fusion_c4_load \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/profile_qkv_norm_rope_vnorm_kvcache_fusion_gemma4_nsys.sh
```

### Part 4 Sustained Load For `nsys`

Run this in a second shell after the Part 4 `nsys` server is ready:

```bash
cd ~/gpu-assignment
RUN_NAME=qkv_norm_rope_vnorm_kvcache_fusion_c4_load \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_aiperf_c4_load.sh
```

### Process Baseline `nsys` Report

```bash
cd ~/gpu-assignment
TRACE_NAME=baseline \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/process_nsys_report.sh
```

### Process Part 4 `nsys` Report

```bash
cd ~/gpu-assignment
TRACE_NAME=qkv-norm-rope-vnorm-kvcache-fusion \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/process_nsys_report.sh
```

### Part 4 Microbenchmark

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_microbenchmark.sh
```

### Part 4 Microbenchmark Without Correctness Check

```bash
cd ~/gpu-assignment
SKIP_CORRECTNESS_CHECK=1 \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/run_microbenchmark.sh
```

### Baseline Compiler Dump

```bash
cd ~/gpu-assignment
PROVIDER=baseline_compiled \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/dump_microbenchmark_compiled_code.sh
```

### Part 4 Fused Compiler Dump

```bash
cd ~/gpu-assignment
PROVIDER=post_gemm_kvcache_compiled \
gpu-assignment/scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/dump_microbenchmark_compiled_code.sh
```

## Results

This section is intentionally left as a placeholder until the first Part 4
runs are collected.

When results are added, record:

1. the microbenchmark CSV and plot summary
2. the compiler-dump proof that the new op replaced the unfused subgraph
3. the matched `nsys` window showing whether `reshape_and_cache_kernel_flash`
   disappeared
4. the end-to-end `AIPerf` comparison against baseline
