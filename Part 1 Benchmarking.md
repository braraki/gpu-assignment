# Part 1 Benchmarking

This document sets up the first benchmarking run for the assignment.

The goal of part 1 is narrow:

1. run vanilla `vllm` with `google/gemma-4-E2B-it`
2. keep the server configuration fixed at concurrency `4`
3. collect one `nsys` trace from the server itself
4. drive a sustained `AIPerf` load at concurrency `4` during the capture window
5. save the commands, artifact locations, and a place to record results

This part does not compare kernel experiments. It is the baseline setup on unmodified `vllm`.

## Target Setup

- model: `google/gemma-4-E2B-it`
- server: vanilla `vllm serve`
- GPU count: `1`
- server-side concurrency cap: `--max-num-seqs 4`
- client load shape: `AIPerf` chat benchmark shape at `concurrency=4`
- profiler: `nsys`

## Files Added For This Part

The script set for this benchmark lives in [gpu-assignment/scripts/part1_benchmarking](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking):

- [serve_vanilla_gemma4.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/serve_vanilla_gemma4.sh): runs the vanilla server without `nsys`
- [profile_vanilla_gemma4_nsys.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh): launches the server under `nsys`
- [run_aiperf_c4_load.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/run_aiperf_c4_load.sh): drives the `AIPerf`-shaped concurrency-4 load

## Why This Setup

The important detail is that `nsys` must wrap the server process, not the client process. The GPU work you care about happens inside `vllm serve`, so the profiler has to be attached there.

The important choice here is to use `AIPerf`, not a custom one-shot client.

That makes the trace much closer to the normal benchmarking path because it preserves the same:

- endpoint type: `chat`
- streaming behavior
- synthetic input length: `512`
- synthetic output length target: `128`
- extra generation control: `ignore_eos:true`
- random seed: `0`
- concurrency control: `4`

So the load is synthetic but reproducible, not fresh unseeded randomness on every run.

The only meaningful change from the normal concurrency sweep is that this part fixes concurrency at `4` and uses one sustained run to keep the server busy during the profiling window.

## Prerequisites

Confirm these are available on the machine where you will run the experiment:

```bash
cd ~/vllm
source .venv/bin/activate
python -c "import vllm; print(vllm.__version__)"
nsys --version
```

If you changed `vllm` source code since the last build:

```bash
cd ~/vllm
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

## Artifact Layout

By default, the scripts write to:

```text
~/gpu-assignment-results/part1-benchmarking/
```

Expected outputs:

- `nsys/vanilla_gemma4_e2b_c4_aiperf_like.nsys-rep`
- `nsys/vanilla_gemma4_e2b_c4_aiperf_like.qdrep` if your local `nsys` version emits it
- `load/vanilla_gemma4_e2b_c4_aiperf_like/`

## Optional Sanity Check Without Profiling

Use this if you want to confirm the server comes up before running `nsys`:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/serve_vanilla_gemma4.sh
```

In another shell:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/run_aiperf_c4_load.sh
```

## Main Profiling Run

Use two shells.

### Shell 1: Start The Server Under `nsys`

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh
```

Default behavior:

- clears `~/.cache/vllm/torch_compile_cache`
- sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- enables `VLLM_NVTX_SCOPES_FOR_PROFILING=1`
- enables `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`
- traces `cuda,nvtx,osrt`
- enables CPU sampling with `--sample=process-tree`
- enables CPU context switch capture with `--cpuctxsw=process-tree`
- waits `140` seconds before capture
- records `30` seconds of trace time

Those defaults assume the server takes about `80` seconds to become ready and leave about `60` additional seconds for post-startup warm-up before the profiler starts recording.

This is now a richer trace than the original low-overhead `cuda,nvtx` pass. The goal is to capture:

- GPU kernels and CUDA API activity
- explicit CUDA memcpy and memset activity
- NVTX scopes from `vllm`
- OS runtime behavior
- CPU sampling and thread scheduling information

### Shell 2: Start The Concurrency-4 Benchmark Load

After the server is healthy, and before the `nsys` delay window expires, run:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/run_aiperf_c4_load.sh
```

Default load behavior:

- waits for `http://localhost:8000/v1/models` before starting
- concurrency: `4`
- request count: `8192`
- warmup request count: `16`
- synthetic input tokens mean: `512`
- synthetic input tokens stddev: `0`
- output tokens mean: `128`
- output tokens stddev: `0`
- extra inputs: `ignore_eos:true`
- random seed: `0`
- endpoint: `/v1/chat/completions`
- streaming: enabled
- output artifact directory: `~/gpu-assignment-results/part1-benchmarking/load/vanilla_gemma4_e2b_c4_aiperf_like`

This is intentionally close to the step-4 concurrency benchmark settings. The default `request_count` is much larger than the step-4 sweep so the load is very likely to stay active for the entire warm-up and `nsys` capture window.

## Useful Overrides

Shorter warm-up:

```bash
cd ~/gpu-assignment/gpu-assignment
WARMUP_SECONDS=110 CAPTURE_SECONDS=20 \
scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh
```

Different port:

```bash
cd ~/gpu-assignment/gpu-assignment
PORT=8001 BASE_URL=http://localhost:8001 \
scripts/part1_benchmarking/run_aiperf_c4_load.sh
```

Different output name:

```bash
cd ~/gpu-assignment/gpu-assignment
TRACE_NAME=vanilla_trial_2 \
scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh
```

Fall back to the lighter trace shape if needed:

```bash
cd ~/gpu-assignment/gpu-assignment
NSYS_TRACE=cuda,nvtx NSYS_SAMPLE=none NSYS_CPUCTXSW=none \
scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh
```

For a matching custom load artifact name, run shell 2 with:

```bash
cd ~/gpu-assignment/gpu-assignment
RUN_NAME=vanilla_trial_2 \
scripts/part1_benchmarking/run_aiperf_c4_load.sh
```

If you want to keep the same benchmark shape but make the load longer or shorter:

```bash
cd ~/gpu-assignment/gpu-assignment
REQUEST_COUNT=4096 WARMUP_REQUEST_COUNT=16 \
scripts/part1_benchmarking/run_aiperf_c4_load.sh
```

## What To Record After The Run

Fill this section in after you collect artifacts.

## Post-Process The `nsys` Trace

After the run completes, generate a few CLI summaries from the saved trace.

Set the trace path once:

```bash
TRACE=~/gpu-assignment-results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like.nsys-rep
OUT=~/gpu-assignment-results/part1-benchmarking/nsys/vanilla_gemma4_e2b_c4_aiperf_like_stats
```

Generate the default text summary:

```bash
nsys stats "$TRACE" > "${OUT}.txt"
```

Generate a CSV-formatted summary:

```bash
nsys stats --format csv "$TRACE" > "${OUT}.csv"
```

If you want specific built-in reports, export them individually:

```bash
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
  "$TRACE" > "${OUT}_focused.txt"
```

Those reports are useful for:

- `cuda_api_sum`: host-side CUDA API time
- `cuda_gpu_kern_sum`: GPU kernel time by kernel name
- `cuda_gpu_mem_time_sum`: CUDA memcpy and memset time
- `osrt_sum`: OS runtime behavior and host-side waits

You can also export SQLite if you want to do deeper custom analysis later:

```bash
nsys export --type sqlite --output "${OUT}" "$TRACE"
```

That should create:

- `${OUT}.txt`
- `${OUT}.csv`
- `${OUT}_focused.txt`
- `${OUT}.sqlite`

### Environment

- machine: 
- GPU: Nvidia L4
- CUDA driver: 580.126.16 (v13.0)
- `vllm` commit: 519df1faec350f517368cea1636415e4ebac9f76
- `nsys` version: 2025.3.2.474-253236389321v0
- date: 4/18/2026

### Server Config Used

- `PORT`:
- `MODEL`:
- `max-model-len`:
- `max-num-seqs`:
- `max-num-batched-tokens`:
- `gpu-memory-utilization`:
- `compilation-config`:

### Load Config Used

- concurrency:
- request count:
- warmup request count:
- synthetic input tokens:
- synthetic output tokens:
- extra inputs:
- random seed:
- streaming:

### Artifact Paths

- `nsys` trace:
- `AIPerf` artifact directory:
- any exported `nsys stats` output:

### Observations

- startup behavior:
- trace quality:
- obvious GPU idle regions:
- obvious synchronization points:
- follow-up questions:

## Initial Analysis

This section summarizes the first pass over:

- the rich `nsys` timeline
- `vanilla_gemma4_e2b_c4_aiperf_like_stats.txt`

### High-Level Read

The vanilla `vllm` run at concurrency `4` looks primarily compute-bound, not transfer-bound.

The strongest signals are:

- GPU time is dominated by a small number of CUTLASS BF16 GEMM kernels
- host-side CUDA API time is dominated by `cudaEventSynchronize`
- explicit host-to-device transfer volume is very small
- there is a repeated cluster of small Triton kernels around norm / GELU / mul style work

### Dominant GPU Kernels

From `cuda_gpu_kern_sum`, the two biggest kernels are:

- CUTLASS BF16 GEMM `...128x2...`: `63.6%`
- CUTLASS BF16 GEMM `...128x1...`: `18.7%`

Together they account for about `82.3%` of total GPU kernel time in this trace.

This means the baseline run is overwhelmingly dominated by dense GEMM compute, which is expected for vanilla Gemma serving.

The next tier is much smaller:

- `kernel_unified_attention_3d`: `2.3%`
- CUTLASS BF16 GEMM with ReLU: `2.3%`
- `tensor_kernel_scan_innermost_dim<float, std::plus<float>>`: `1.8%`
- `cunn_SoftMaxForward`: `1.1%`

### Host-Side Behavior

From `cuda_api_sum`, `cudaEventSynchronize` accounts for:

- `92.5%` of CUDA API time
- `29.39s` total
- `2574` calls
- `11.4 ms` average per call

This means host-side CUDA API time is dominated by waiting for GPU events, not by kernel launch overhead.

Important interpretation:

- this does **not** automatically mean the run is bottlenecked by CPU overhead
- it more likely means the CPU often waits for GPU work to complete
- it becomes a serious optimization target only when those syncs line up with real GPU bubbles in the timeline

In the zoomed regions inspected so far, `cudaEventSynchronize` often appears as part of the control loop, but not always as the cause of a large GPU idle gap.

### Memory Operation Summary

From `cuda_gpu_mem_time_sum` and `cuda_gpu_mem_size_sum`:

- H2D total size is only about `16.1 MB`
- D2H total size is negligible
- D2D total size is much larger at about `5310 MB`
- memsets and H2D copies are numerous, but individually very small

Interpretation:

- this run does **not** look host-to-device-transfer-bound
- the H2D copies are many tiny control/data movements, not large payload transfers
- the more substantial memory-motion category is device-to-device traffic, not PCIe traffic

### Repeated Small-Kernel Cluster

Several Triton kernels recur very frequently:

- `triton_red_fused_add_rms_norm_0`: `0.6%`, `45036` instances
- `triton_poi_fused_gelu_mul_slice_1`: `0.6%`, `45036` instances
- `triton_red_fused_add_mul_rms_norm_4`: `0.6%`, `45036` instances
- `triton_red_fused_add_rms_norm_2`: `0.4%`, `45036` instances
- `triton_poi_fused_gelu_mul_3`: `0.2%`, `45036` instances

Individually these are small, but together they form a repeated glue-kernel cluster that is more realistic to optimize than the large CUTLASS GEMMs.

### Additional Follow-Up Candidate

The scan and sort path is large enough to notice:

- `tensor_kernel_scan_innermost_dim<float, std::plus<float>>`: `1.8%`
- `DeviceRadixSortOnesweepKernel`: `0.6%`
- plus smaller radix-sort helpers below that

This path is worth investigating if it is on the steady-state decode path rather than only occasional housekeeping.

### What Seems Less Important

The first pass does **not** suggest starting with:

- large H2D transfer optimization
- generic CPU overload debugging
- trying to out-optimize the dominant CUTLASS GEMMs

Those are not where the current trace suggests the easiest wins are.

### Best Optimization Targets To Investigate Next

The most promising next targets are:

1. the repeated Triton norm / GELU / mul fusion chain
2. the scan + radix-sort path
3. `cudaEventSynchronize` only where it clearly correlates with visible GPU bubbles

## Next Step

After the first trace is saved, the next useful follow-up is:

1. run `nsys stats` on the saved `.nsys-rep`
2. inspect the trace in the Nsight Systems GUI
3. decide whether the first `c=4` load window is enough or whether a second run should use a longer capture window or a larger `request_count`
