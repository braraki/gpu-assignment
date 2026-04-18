# Part 2 Decoder Fusion

This document captures the next optimization direction after the vanilla
benchmarking pass: **Gemma 4 decoder residual fusion**.

The target is narrow and source-level. Instead of trying to optimize a
compiler-generated Triton kernel name directly, the experiment replaces an
explicit residual add followed by RMSNorm with vLLM's existing fused
add+RMSNorm primitive in the Gemma 4 decoder hot path.

## What The Experiment Is

This part evaluates the `decoder-residual-fusion` Gemma 4 experiment mode.

The experiment lives in the Gemma 4 decoder implementation:

- experiment switch:
  [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py:459)
- NVTX scope names:
  [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py:73)

The relevant flag is:

```python
self.use_decoder_residual_fusion = (
    kernel_experiment == "decoder-residual-fusion"
)
```

This is a different experiment from:

- Q/K RMSNorm + RoPE fusion
- PLE GELU x mul fusion
- async output sync reduction

Those target different parts of the model or serving path.

## Where It Occurs In Code

The targeted transition is inside `Gemma4DecoderLayer.forward`:

- decoder layer body:
  [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py:631)
- pre-FF residual/norm transition:
  [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py:649)

Baseline path:

```python
hidden_states = self.post_attention_layernorm(hidden_states)

with scope_context:
    hidden_states = hidden_states + residual
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
```

Fusion path:

```python
hidden_states = self.post_attention_layernorm(hidden_states)

with scope_context:
    hidden_states, residual = self.pre_feedforward_layernorm(
        hidden_states, residual
    )
```

This is the decoder transition between:

1. attention output
2. post-attention RMSNorm
3. residual add
4. pre-feedforward RMSNorm
5. MLP input

So the optimization target is the **pre-FF residual handoff**, not the MLP
itself and not the main attention kernels.

## The Fused Primitive Being Reused

This experiment does not start with a new Triton kernel. It reuses vLLM's
existing fused add+RMSNorm path.

- Python wrapper:
  [layernorm.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/layers/layernorm.py:52)
- CUDA entry point:
  [layernorm_kernels.cu](/Users/brandonaraki/projects/gpu-assignment/vllm/csrc/layernorm_kernels.cu:239)

The wrapper is:

```python
def fused_add_rms_norm(x, residual, weight, variance_epsilon):
    ops.fused_add_rms_norm(x, residual, weight, variance_epsilon)
    return x, residual
```

The CUDA side takes:

- `input` shaped like `[..., hidden_size]`
- `residual` shaped like `[..., hidden_size]`
- `weight` shaped like `[hidden_size]`

## Why This Is The Right Abstraction

The baseline trace contains repeated Triton kernels in the
`triton_red_fused_add_rms_norm_*` family, but those names are compiler
artifacts, not stable source-level targets.

So the optimization target should be:

- the Gemma 4 decoder residual transition

not:

- a specific generated kernel such as `triton_red_fused_add_rms_norm_0`

Why this is better:

- the source pattern is stable across recompiles
- Triton suffixes like `_0` and `_2` are not semantically reliable
- one source-level change can reduce an entire generated-kernel family
- vLLM already has the fused primitive needed for the experiment

## Baseline Evidence From Part 1

Part 2 does not repeat baseline benchmarking or baseline `nsys` capture. Use
Part 1 as the baseline reference:

- [Part 1 Benchmarking.md](</Users/brandonaraki/projects/gpu-assignment/gpu-assignment/Part 1 Benchmarking.md>)

The baseline stats file from Part 1 is:

- [vanilla_gemma4_e2b_c4_aiperf_like_stats.txt](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:1)

The main run is still dominated by dense GEMM compute:

- main CUTLASS BF16 GEMM:
  `63.6%`, `18,907,321,403 ns`, `167,065` instances
  [stats](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:80)
- second CUTLASS BF16 GEMM:
  `18.7%`, `5,549,157,316 ns`, `97,285` instances
  [stats](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:81)

The repeated add+RMSNorm-style Triton family is present in the hot path:

- `triton_red_fused_add_rms_norm_0`
  - `0.6%`
  - `181,374,949 ns`
  - `45,036` instances
  - `4.03 us` average
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:89)
- `triton_red_fused_add_rms_norm_2`
  - `0.4%`
  - `125,175,707 ns`
  - `45,036` instances
  - `2.78 us` average
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:94)

There is also host-side evidence that the run is generally paced by GPU work:

- `cudaEventSynchronize`
  - `92.5%` of CUDA API time
  - `29,386,739,541 ns`
  - `2,574` calls
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part-1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:47)

So this experiment is not trying to beat the main GEMMs. The more realistic
goal is to make the decoder pre-FF handoff cheaper and cleaner.

## Why This Could Improve Performance

In the baseline decoder path, the transition materializes:

1. `hidden_states + residual`
2. store the updated tensor
3. set `residual = hidden_states`
4. run RMSNorm over the result

The fused primitive can express that transition directly:

- compute the residual add and normalization in one specialized path
- thread `residual` explicitly
- reduce intermediate materialization
- reduce small-kernel and launch overhead around the pre-FF handoff

This is one of the cases where a custom vLLM op can plausibly outperform
default `torch.compile`:

- the op has stronger semantics than a generic compiler fusion
- it owns both `hidden_states` and `residual`
- it already exists in vLLM, so the experiment risk is lower than writing a
  new kernel from scratch

## What We Expect To Change

If decoder residual fusion works as intended, the expected changes are:

### In The Trace

- fewer small kernel boundaries in the decoder pre-FF transition
- reduced presence of `triton_red_fused_add_rms_norm_*` kernels in the
  affected scope
- no regression in the adjacent dominant GEMM kernels
- no new GPU bubbles introduced by the fused path

With NVTX scopes enabled, the relevant scope names are:

- `gemma4.decoder.pre_ff_residual_norm:baseline`
- `gemma4.decoder.pre_ff_residual_norm:decoder_residual_fusion`

Those are defined in:

- [gemma4.py](/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/model_executor/models/gemma4.py:73)

### In `nsys stats`

We would expect:

- lower total time for the add+RMSNorm-style Triton family in the affected path
- potentially fewer small launches around the decoder transition
- no change in the dominant GEMM ordering

### In End-To-End Serving

The expected win is modest:

- slightly higher throughput at the same concurrency
- or at minimum no regression with a cleaner decoder transition

This is not expected to be a large end-to-end improvement because the run
remains dominated by large GEMM kernels. The case for the experiment is that it
is:

- local
- low-risk
- source-level
- already supported by an existing vLLM fused primitive

## Validation Plan

To validate this experiment cleanly:

1. use Part 1 as the baseline benchmark and baseline `nsys` reference
2. run the `decoder-residual-fusion` `AIPerf` sweep
3. collect one matching rich `nsys` trace for `decoder-residual-fusion`
4. compare the `gemma4.decoder.pre_ff_residual_norm:*` NVTX scopes against the
   Part 1 baseline trace
5. compare the add+RMSNorm Triton family and nearby launch behavior
6. confirm there is no throughput regression relative to Part 1

## Scripts

The Part 2 harness now follows the same structure as Part 1. The scripts live
in:

- [scripts/part2_decoder_fusion](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion)

Available scripts:

- [serve_decoder_residual_fusion_gemma4.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion/serve_decoder_residual_fusion_gemma4.sh)
- [run_aiperf_sweep.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion/run_aiperf_sweep.sh)
- [run_aiperf_c4_load.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion/run_aiperf_c4_load.sh)
- [profile_decoder_residual_fusion_gemma4_nsys.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion/profile_decoder_residual_fusion_gemma4_nsys.sh)
- [process_nsys_report.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part2_decoder_fusion/process_nsys_report.sh)

## Exact Commands

These assume you are starting from `~/gpu-assignment/gpu-assignment`.

### Benchmark Sweep

Baseline benchmark results come from Part 1. For the Part 2 fusion run:

```bash
scripts/part2_decoder_fusion/serve_decoder_residual_fusion_gemma4.sh
```

In a second terminal:

```bash
cd ~/gpu-assignment/gpu-assignment
RUN_SET=decoder-residual-fusion scripts/part2_decoder_fusion/run_aiperf_sweep.sh
```

### `nsys` Capture

Baseline `nsys` trace comes from Part 1. For the Part 2 fusion trace:

```bash
scripts/part2_decoder_fusion/profile_decoder_residual_fusion_gemma4_nsys.sh
```

In a second terminal:

```bash
cd ~/gpu-assignment/gpu-assignment
RUN_NAME=decoder_residual_fusion_c4_load scripts/part2_decoder_fusion/run_aiperf_c4_load.sh
```

Process the report:

```bash
cd ~/gpu-assignment/gpu-assignment
TRACE_NAME=decoder_residual_fusion scripts/part2_decoder_fusion/process_nsys_report.sh
```

## What This Experiment Is Not

This experiment is not:

- Q/K norm + RoPE fusion
- a `head_dim=512` attention-kernel experiment
- a custom replacement for the dominant CUTLASS GEMMs

That distinction matters because the earlier unsupported-`512` failure and the
large regression from a custom `512` kernel came from the Q/K RMSNorm + RoPE
path, not this decoder residual path.
