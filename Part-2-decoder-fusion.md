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

If you need to rerun the baseline sweep locally, Part 1 now includes:

- [run_aiperf_sweep.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/run_aiperf_sweep.sh)

The baseline stats file from Part 1 is:

- [vanilla_gemma4_e2b_c4_aiperf_like_stats.txt](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:1)

The main run is still dominated by dense GEMM compute:

- main CUTLASS BF16 GEMM:
  `63.6%`, `18,907,321,403 ns`, `167,065` instances
  [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:80)
- second CUTLASS BF16 GEMM:
  `18.7%`, `5,549,157,316 ns`, `97,285` instances
  [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:81)

The repeated add+RMSNorm-style Triton family is present in the hot path:

- `triton_red_fused_add_rms_norm_0`
  - `0.6%`
  - `181,374,949 ns`
  - `45,036` instances
  - `4.03 us` average
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:89)
- `triton_red_fused_add_rms_norm_2`
  - `0.4%`
  - `125,175,707 ns`
  - `45,036` instances
  - `2.78 us` average
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:94)

There is also host-side evidence that the run is generally paced by GPU work:

- `cudaEventSynchronize`
  - `92.5%` of CUDA API time
  - `29,386,739,541 ns`
  - `2,574` calls
  - [stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:47)

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

## Operator Microbenchmark

Part 2 now includes an operator-level microbenchmark for the exact pre-FF
residual handoff above. This benchmark ignores `nsys` and serving-path
profiling and instead compares the isolated operator sequence directly.

The benchmark lives in:

- [benchmark_decoder_residual_fusion.py](/Users/brandonaraki/projects/gpu-assignment/vllm/benchmarks/kernels/benchmark_decoder_residual_fusion.py)

It compares these providers:

- `baseline_eager`
  - explicit `hidden_states + residual`, then RMSNorm
- `baseline_compiled`
  - the same baseline operator sequence wrapped in `torch.compile`
- `fusion_compiled`
  - the fused add+RMSNorm path wrapped in `torch.compile`
- `fusion_custom_op`
  - vLLM's existing `fused_add_rms_norm(...)` path

By default it:

- derives `hidden_size` and `rms_norm_eps` from `google/gemma-4-E2B-it`
- uses `bfloat16`
- benchmarks flattened token counts `1 4 16 64 256 1024`
- runs a correctness check that compares both:
  - normalized `hidden_states`
  - updated `residual`

### Run It

Use the Part 2 script:

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/part2_decoder_fusion/run_microbenchmark.sh
```

Outputs land under:

- [results/part2-decoder-fusion/microbench](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/microbench)

The main CSV is:

- [decoder_residual_fusion_benchmark.csv](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/microbench/decoder_residual_fusion_benchmark.csv)

The script also writes one latency-vs-token-count plot per hidden size.

### Dump Compiler-Generated Code

To prove which generated kernels correspond to the isolated changed block, run
the microbenchmark with compiler code dumping enabled for each compiled mode.

Baseline compiled dump:

```bash
cd ~/gpu-assignment
PROVIDER=baseline_compiled \
gpu-assignment/scripts/part2_decoder_fusion/dump_microbenchmark_compiled_code.sh
```

Fusion compiled dump:

```bash
cd ~/gpu-assignment
PROVIDER=fusion_compiled \
gpu-assignment/scripts/part2_decoder_fusion/dump_microbenchmark_compiled_code.sh
```

By default this profiles the isolated operator at `1024` tokens and writes all
artifacts under:

- [results/part2-decoder-fusion/microbench/compiler_dumps](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/microbench/compiler_dumps)

Each run directory contains:

- `torch_output_code.log`
  - stdout/stderr from `TORCH_LOGS=output_code,recompiles,graph_breaks`
- `benchmark_outputs/`
  - the one-provider benchmark CSV and plot
- `vllm_cache/torch_compile_cache/`
  - the saved Dynamo and Inductor-generated code, using
    `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked`

What to compare:

1. The baseline-compiled dump should lower the explicit add-then-RMSNorm
   sequence from the unfused source form.
2. The fusion-compiled dump should lower the `fused_add_rms_norm(...)` path.
3. The kernel names and Triton code shapes observed there can then be matched
   back to the full-server trace with much stronger confidence than kernel-name
   guessing alone.

Helpful overrides:

- `NUM_TOKENS="256 1024"` to dump more than one token count
- `HIDDEN_SIZES="1536"` to force a specific hidden size
- `EPS="1e-6"` to override the model-derived RMSNorm epsilon

### How To Interpret A Win

The main comparison of interest is:

- `baseline_compiled`
- `fusion_custom_op`

This benchmark is successful if `fusion_custom_op` shows lower median latency
than `baseline_compiled` at Gemma4-relevant token counts without failing the
correctness check.

### Results

The downloaded microbenchmark results are in:

- [decoder_residual_fusion_benchmark.csv](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/microbench/decoder_residual_fusion_benchmark.csv)
- [decoder_residual_fusion_hidden_size_1536.png](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/microbench/decoder_residual_fusion_hidden_size_1536.png)

This run used:

- `hidden_size=1536`
- `dtype=bfloat16`
- token counts `1 4 16 64 256 1024`

The main comparison, `baseline_compiled` vs `fusion_custom_op`, shows a clear
size-dependent crossover:

| Tokens | `baseline_compiled` ms | `fusion_custom_op` ms | Result |
| --- | ---: | ---: | --- |
| 1 | 0.003282 | 0.003527 | fusion slower by 7.5% |
| 4 | 0.003562 | 0.003601 | fusion slower by 1.1% |
| 16 | 0.003732 | 0.003788 | fusion slower by 1.5% |
| 64 | 0.003989 | 0.004867 | fusion slower by 22.0% |
| 256 | 0.006464 | 0.006269 | fusion faster by 3.0% |
| 1024 | 0.019494 | 0.016163 | fusion faster by 17.1% |

What jumps out:

- `baseline_eager` is much slower everywhere, so `torch.compile` is already
  doing a strong job on this operator sequence.
- The fused custom op does **not** win uniformly. It loses at small token
  counts and only starts to pull ahead once the workload is large enough.
- The strongest win in this dataset is at `1024` tokens, where the fused path
  is about `17%` faster than the compiled baseline.
- The worst point is `64` tokens, where the fused path is about `22%` slower
  than the compiled baseline.

Interpretation:

- The custom fused add+RMSNorm path appears to have a higher fixed-cost regime
  than the compiled baseline, which hurts it on small inputs.
- At larger token counts, the fused path begins to amortize that overhead and
  outperform the compiled baseline.
- So this experiment currently looks more promising for larger flattened token
  batches than for latency-sensitive tiny batches.

Practical takeaway:

- This is **not** a blanket replacement win over `torch.compile`.
- It is a conditional win that depends on workload size.
- If we continue this direction, the next useful step is to compare these
  token-count breakpoints against the actual serving shapes we hit most often
  in Gemma4 production traces and benchmarks.

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

## Results

The current Part 2 result set includes:

- Part 1 baseline `nsys stats` output:
  [vanilla_gemma4_e2b_c4_aiperf_like_stats.txt](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:1)
- Part 2 fusion `nsys stats` output:
  [decoder_residual_fusion_stats.txt](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/nsys/decoder_residual_fusion_stats.txt:1)
- fusion `AIPerf` artifact from the profiling run:
  [profile_export_aiperf.json](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/load/profile_export_aiperf.json:1)

### Main Result

From the refreshed Part 1 and Part 2 traces, the result is still: **no clear
macro-level win**.

### High-Level Runner Impact

At the high-level NVTX layer, the forward path did not improve. There is about a `2.0%` regression in the average forward NVTX span.

Some adjacent scopes improved slightly:

- `preprocess`: about `-0.6%`
- `sample`: about `-3.3%`
- `postprocess`: about `-3.6%`
- `schedule: allocate_slots`: about `-2.8%`

But those small wins were not enough to offset the forward-path regression.

### Small Kernel Family Changes

The strongest full-trace kernel clue is now the `_0` family only.

From the full server traces:

- baseline:
  - `triton_red_fused_add_rms_norm_0`
  - `210,266,703 ns`
  - [baseline stats](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats.txt:100)
- fusion:
  - `triton_red_fused__to_copy_add_mean_mul_pow_rms_norm_rsqrt_0`
  - `180,364,094 ns`
  - [fusion stats](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/nsys/decoder_residual_fusion_stats.txt:102)

That is a reduction of about `14.2%` in total kernel time for the most likely
affected kernel motif.

The reason we can now tie this kernel family to the changed source block is the
isolated compiler dump from the operator microbenchmark.

In the isolated **baseline compiled** microbenchmark, the dumped Inductor graph
for the exact changed block shows:

- source ops:
  - `aten.add`
  - `_to_copy`
  - `aten.pow`
  - `aten.mean`
  - `aten.rsqrt`
  - `aten.mul`
  - [baseline compiled dump](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/baseline_compiled_tokens1024/torch_output_code.log:42)
- generated Triton kernel:
  - `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0`
  - [baseline compiled dump](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/baseline_compiled_tokens1024/torch_output_code.log:69)
- launch site:
  - [baseline compiled dump](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/baseline_compiled_tokens1024/torch_output_code.log:158)

That is exactly the unfused source pattern we changed in `gemma4.py`:

1. residual add
2. type conversion
3. RMSNorm reduction math
4. weight multiply

So the isolated compiler dump proves that this kernel shape is a valid compiled
form of the baseline decoder residual handoff.

In the isolated **fusion compiled** microbenchmark, the changed block does not
lower into another RMSNorm Triton reduction kernel. Instead, the dump shows a
direct custom-op call:

- `torch.ops._C.fused_add_rms_norm.default(...)`
  [fusion compiled dump](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/fusion_compiled_tokens1024/torch_output_code.log:109)

with only clone helper Triton kernels around it:

- `triton_poi_fused_as_strided_clone_0`
  [fusion compiled dump](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/fusion_compiled_tokens1024/torch_output_code.log:50)

That means the compiler-dump proof is strongest in one direction:

- it proves that the baseline source block can compile into the
  `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` family
- it does **not** justify treating every RMSNorm-family kernel in the fusion
  server trace as the decoder-fusion replacement

Because of that, I no longer treat these as decoder-fusion evidence:

- `triton_red_fused_add_mul_rms_norm_4`
- `triton_red_fused_add_rms_norm_2`
- `triton_red_fused__to_copy_add_rms_norm_2`

Those kernels may come from other unchanged decoder norm sites and should not
be used as primary proof for this experiment.

### End-To-End Benchmark Caveat

There is a saved fusion `AIPerf` artifact from the profiling run, but it should
not be treated as a clean benchmark result because it was collected during the
`nsys` capture workflow and includes many failed requests when the server was
terminated at the end of the profiling window.

Fusion artifact:

- output token throughput: `169.95 tok/s`
- output token throughput per user: `44.41 tok/s/user`
- TTFT: `146.00 ms`
- inter-token latency: `22.52 ms`
- request latency: `2972.15 ms`
- error request count: `450`
  [fusion artifact](/Users/brandonaraki/projects/gpu-assignment/results/part2-decoder-fusion/load/profile_export_aiperf.json:1)

For reference, the comparable Part 1 profiling-window load artifact is also not
clean:

- output token throughput: `169.36 tok/s`
- output token throughput per user: `44.28 tok/s/user`
- TTFT: `150.41 ms`
- inter-token latency: `22.59 ms`
- request latency: `2894.53 ms`
- error request count: `217`
  [Part 1 artifact](/Users/brandonaraki/projects/gpu-assignment/results/part1-benchmarking/load/vanilla_gemma4_e2b_c4_aiperf_like/profile_export_aiperf.json:1)

On these profiling-window artifacts, fusion is basically flat:

- output token throughput: about `+0.35%`
- output token throughput per user: about `+0.29%`
- TTFT: about `-2.9%`
- inter-token latency: about `-0.3%`
- request latency: about `+2.7%`

The problem is that the failure counts are still nontrivial and now worse in
the fusion run (`217` baseline vs `450` fusion). So these are still not clean
benchmark-quality end-to-end numbers.

### Interpretation

The current result is:

- the generated add+RMSNorm kernel family changed and got a modest aggregate
  reduction
- no, it did not change the dominant GEMM bottleneck structure
- no, it did not meaningfully change `cudaEventSynchronize`
- no, the average `gpu_model_runner: forward` NVTX span did not improve
- no, the saved profiling-window `AIPerf` artifact does not show a clean,
  trustworthy throughput win
- no, the saved Part 1 and Part 2 exports still do not contain the decoder
  `gemma4.decoder.pre_ff_residual_norm:*` scope needed for a direct targeted
  measurement

So the right write-up is:

- **the baseline decoder residual block maps directly to a specific compiled kernel family**
- **that kernel family is cheaper in the full fusion server trace**
- **macro-level improvement not demonstrated from these saved logs**

The next step is to run a clean fusion `AIPerf` sweep, outside the forced
`nsys` shutdown path, and if you want scoped proof, rerun the fusion trace with
the custom Gemma4 NVTX scopes verified in the saved export.

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
