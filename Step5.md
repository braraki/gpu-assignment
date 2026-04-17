# Step 5: Collect And Inspect Nsight Systems Traces

This document walks through step 5 of the assignment:

1. collect GPU traces with `nsys`
2. intentionally force a synchronization point
3. compare the baseline trace against the forced-sync trace
4. use the traces to identify GPU bubbles, synchronization, and scheduling gaps

This is written for the current setup:

- model: `google/gemma-4-E2B-it`
- single GPU: `g6.xlarge`
- `vllm` already runs on the EC2 instance
- `AIPerf` is already available for generating load

## Goal

You want two traces:

- a baseline trace of the normal server path
- a second trace where you intentionally add a synchronization point, for example `t.sum().item()`

That comparison makes it easier to see:

- where the GPU is busy
- where the GPU is idle
- whether there are host-side stalls
- whether the forced sync creates obvious bubbles or serialization

## Prerequisites

On the EC2 instance, confirm:

```bash
nsys --version
```

Also confirm the server can already run:

```bash
cd ~/vllm
source .venv/bin/activate
python -c "import vllm; print(vllm.__version__)"
```

Create a results directory:

```bash
mkdir -p ~/gpu-assignment-results/step5-nsys
```

## Stability Note For `g6.xlarge`

Do not treat step 5 like a normal benchmark run.

On a `g6.xlarge`, you only have:

- `1x` NVIDIA L4 GPU
- `4` vCPUs
- `16 GiB` of system RAM

That is enough to run the model, but it is not much headroom once you stack:

- the `vllm` server
- `nsys` trace collection
- `AIPerf`
- trace-file writes

If the machine freezes and you lose SSH, that usually means the host became resource-starved hard enough that the instance stopped servicing interactive work. In practice, that is commonly:

- host RAM pressure or kernel OOM
- severe CPU contention on a `4` vCPU machine
- less commonly, a driver-level hang

For this step, prioritize a smaller, stable trace over a "max-load" trace.

## Recommended Pre-Flight

Before profiling, set:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The local `vllm` profiling docs recommend `spawn` when using `nsys`. Do not skip this on Linux.

Also clear the `vllm` torch compile cache before switching profiling modes or after editing model code:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

Use that before:

- the first baseline trace if you are not sure what was compiled previously
- the forced-sync trace after editing `gemma4_mm.py`
- any rerun where you changed eager mode, CUDA graph settings, or other compile-related flags

This is the cache you most likely want here. It avoids stale compiled artifacts carrying across experiment modes.

## High-Level Plan

Run step 5 in this order:

1. collect a small, stable baseline `.nsys-rep`
2. collect a second `.nsys-rep` with a forced sync point
3. run `nsys stats` for quick summaries
4. optionally run the local `vllm` helper in `tools/profiler/nsys_profile_tools`
5. only after the small run succeeds, consider a slightly heavier rerun
6. download the `.nsys-rep` files and inspect them in the Nsight Systems GUI on your laptop

## Baseline Trace

Stop any already-running server first, then relaunch `vllm` under `nsys`.

On the EC2 instance:

```bash
cd ~/vllm
source .venv/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
rm -rf ~/.cache/vllm/torch_compile_cache

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay 240 \
  --duration 30 \
  --output ~/gpu-assignment-results/step5-nsys/baseline \
  vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --language-model-only \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.70 \
    --enable-chunked-prefill \
    --async-scheduling \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Why these `nsys` flags:

- `--delay 30` gives the server time to start before capture begins
- `--duration 30` keeps the trace short enough to be practical on a small instance
- if the first baseline is too short to capture a clear steady-state window, do a follow-up rerun at `--duration 45`
- `--trace=cuda,nvtx` keeps the first pass lower-overhead
- `--cuda-graph-trace=node` is useful for the graph-enabled baseline
- `--trace-fork-before-exec=true` matches the local `vllm` profiler guidance

Why the smaller server config:

- step 5 is about getting an interpretable trace, not maximizing throughput
- `2048 / 4 / 512 / 0.70` leaves more room for profiler overhead than the step 3 bring-up config
- the trace is still useful even if it is collected at a lighter load

### Generate Load During The Capture Window

In a second SSH session, after starting the command above, run a short AIPerf load.

Start with a deliberately small load first:

```bash
source ~/aiperf-venv/bin/activate

aiperf profile \
  --model google/gemma-4-E2B-it \
  --tokenizer google/gemma-4-E2B-it \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 1 \
  --request-count 32 \
  --warmup-request-count 4 \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 64 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step5-nsys/baseline_load
```

If that works cleanly once, you can try one cautious step up:

- increase `--concurrency` from `1` to `2`
- optionally increase `--request-count` from `32` to `64`

Do not start with concurrency `4` on `g6.xlarge` for the first trace attempt.

If you specifically need host-runtime detail later, rerun after the stable baseline succeeds and change:

```text
--trace=cuda,nvtx
```

to:

```text
--trace=cuda,nvtx,osrt
```

That richer trace is more expensive, so treat it as a second pass, not the default first pass.

When `nsys` exits, you should have:

```text
~/gpu-assignment-results/step5-nsys/baseline.nsys-rep
```

## Optional: Benchmark Decoder Residual Fusion

If you want to use step 5 traces to evaluate the existing
`decoder-residual-fusion` experiment instead of only comparing
baseline vs forced sync, keep the load shape fixed and change only:

- `--gemma4-kernel-experiment baseline`
- `--gemma4-kernel-experiment decoder-residual-fusion`

For this comparison, do **not** inject the forced sync point.

### Why Enable Profiling Scopes

The Gemma 4 decoder path now emits optional profiling scopes around the
transition between:

1. `post_attention_layernorm`
2. residual add or fused residual threading
3. `pre_feedforward_layernorm`

Enable one of these before launching the server:

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

or:

```bash
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
```

In the trace, look for:

- `gemma4.decoder.pre_ff_residual_norm:baseline`
- `gemma4.decoder.pre_ff_residual_norm:decoder_residual_fusion`

That makes it much easier to compare the exact decoder transition that
the experiment changes.

### Collect A Baseline Experiment Trace

Run the baseline experiment mode first:

```bash
cd ~/vllm
source .venv/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
rm -rf ~/.cache/vllm/torch_compile_cache

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay 240 \
  --duration 30 \
  --output ~/gpu-assignment-results/step5-nsys/decoder_residual_baseline \
  vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --language-model-only \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.80 \
    --enable-chunked-prefill \
    --async-scheduling \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --gemma4-kernel-experiment baseline
```

In a second SSH session, generate the load:

```bash
source ~/aiperf-venv/bin/activate

aiperf profile \
  --model google/gemma-4-E2B-it \
  --tokenizer google/gemma-4-E2B-it \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 1 \
  --request-count 32 \
  --warmup-request-count 4 \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 64 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step5-nsys/decoder_residual_baseline_load
```

### Collect A Decoder Residual Fusion Trace

Stop the baseline server, clear the compile cache again, then rerun with
only the experiment flag changed:

```bash
cd ~/vllm
source .venv/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
rm -rf ~/.cache/vllm/torch_compile_cache

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay 240 \
  --duration 30 \
  --output ~/gpu-assignment-results/step5-nsys/decoder_residual_fusion \
  vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --language-model-only \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.80 \
    --enable-chunked-prefill \
    --async-scheduling \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --gemma4-kernel-experiment decoder-residual-fusion
```

Use the same AIPerf load shape again, but write to a separate artifact
directory:

```bash
source ~/aiperf-venv/bin/activate

aiperf profile \
  --model google/gemma-4-E2B-it \
  --tokenizer google/gemma-4-E2B-it \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 1 \
  --request-count 32 \
  --warmup-request-count 4 \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 64 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step5-nsys/decoder_residual_fusion_load
```

This should leave you with:

```text
~/gpu-assignment-results/step5-nsys/decoder_residual_baseline.nsys-rep
~/gpu-assignment-results/step5-nsys/decoder_residual_fusion.nsys-rep
```

### Optional Throughput Sweep

If the small trace succeeds and you also want end-to-end benchmark
numbers, reuse the step-6 AIPerf sweep for concurrencies `1 2 4 8` with
the same two server modes:

- `--gemma4-kernel-experiment baseline`
- `--gemma4-kernel-experiment decoder-residual-fusion`

Store those outputs under separate directories, for example:

- `~/gpu-assignment-results/step6-decoder-residual-fusion/baseline_c1`
- `~/gpu-assignment-results/step6-decoder-residual-fusion/decoder_residual_fusion_c1`

Use the combined Pareto plotting flow from
`Documentation/decoder_residual_fusion.md` when you want throughput and
latency numbers, and use the step-5 traces when you want to validate
that any throughput change matches a real kernel-level improvement.

## Forced-Sync Trace

For the sync experiment, do two things:

1. inject a deliberate sync point
2. turn off CUDA graphs so the experiment is easier to interpret

### Suggested Sync Injection Point

For the current Gemma 4 path, the simplest place is inside:

```text
~/vllm/vllm/model_executor/models/gemma4_mm.py
```

Right after:

```python
hidden_states = self.language_model.model(...)
```

Add a temporary guard like this:

```python
import os
...
hidden_states = self.language_model.model(
    input_ids,
    positions,
    per_layer_inputs=per_layer_inputs,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **kwargs,
)

if os.getenv("VLLM_FORCE_SYNC_POINT") == "1":
    _ = hidden_states.sum().item()

return hidden_states
```

Notes:

- keep this change temporary
- put it on a throwaway branch or revert it after the experiment
- `t.sum().item()` is intentionally bad here; that is the point of the experiment

### Why Disable CUDA Graphs Here

For this experiment, prefer eager execution so the forced sync is easier to spot.

Use:

- `--enforce-eager`

Do not use the graph-heavy compilation config for this run.

### Run The Sync Trace

After adding the guarded sync point, run:

```bash
cd ~/vllm
source .venv/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
rm -rf ~/.cache/vllm/torch_compile_cache

VLLM_FORCE_SYNC_POINT=1 \
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --delay 30 \
  --duration 30 \
  --output ~/gpu-assignment-results/step5-nsys/forced_sync \
  vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --language-model-only \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.70 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enforce-eager
```

Then in the second SSH session, run the same AIPerf load again:

```bash
source ~/aiperf-venv/bin/activate

aiperf profile \
  --model google/gemma-4-E2B-it \
  --tokenizer google/gemma-4-E2B-it \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 1 \
  --request-count 32 \
  --warmup-request-count 4 \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 64 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step5-nsys/forced_sync_load
```

When `nsys` exits, you should have:

```text
~/gpu-assignment-results/step5-nsys/forced_sync.nsys-rep
```

## Quick CLI Analysis

Run `nsys stats` on both traces:

```bash
nsys stats ~/gpu-assignment-results/step5-nsys/baseline.nsys-rep
nsys stats ~/gpu-assignment-results/step5-nsys/forced_sync.nsys-rep
```

This is the fastest way to get a first summary without opening the GUI.

## If The Instance Froze

If the EC2 instance froze badly enough that you lost SSH, do not immediately retry with the same command.

After the instance comes back, check the previous boot logs:

```bash
sudo journalctl -k -b -1 | rg -i 'oom|out of memory|killed process|xid|nvrm'
```

Also check whether the current boot shows GPU-driver complaints:

```bash
sudo dmesg -T | rg -i 'xid|nvrm|oom|out of memory'
```

Interpretation:

- `oom` or `killed process` strongly suggests host memory pressure
- `xid` or `nvrm` points more toward a GPU-driver-level problem

If you see OOM-like messages, reduce load further before retrying:

- `--concurrency 1`
- `--request-count 16`
- `--synthetic-input-tokens-mean 128`
- `--output-tokens-mean 32`
- if needed, lower `--gpu-memory-utilization` again to `0.60`

If even that is unstable, stop and record that `g6.xlarge` is too small for this profiling setup. Move to a larger instance such as `g6.2xlarge` or `g6.4xlarge` for step 5.

## Optional: Use The Local vLLM Nsight Helper

The local `vllm` checkout already has a helper for post-processing `.nsys-rep` files:

```text
~/vllm/tools/profiler/nsys_profile_tools/gputrc2graph.py
```

The corresponding local README is:

```text
~/vllm/tools/profiler/nsys_profile_tools/README.md
```

That tool is useful if you want a quick categorized kernel-level view from the trace files.

## What To Look For

Compare the baseline and forced-sync traces for:

- wider idle gaps between GPU kernels
- host-side gaps before kernel launch
- `cudaMemcpy` or other transfer activity that should not be there
- explicit synchronization behavior caused by `.item()`
- whether decode and scheduling cadence becomes more serialized

In the forced-sync trace, you should expect:

- more obvious GPU bubbles
- less overlap
- more waiting on the CPU side

## Download The Trace Files

From your laptop, use `scp`:

```bash
scp -i <KEY_PATH> \
  ubuntu@<PUBLIC_DNS_OR_IP>:~/gpu-assignment-results/step5-nsys/baseline.nsys-rep \
  ubuntu@<PUBLIC_DNS_OR_IP>:~/gpu-assignment-results/step5-nsys/forced_sync.nsys-rep \
  .
```

Then open those files in the Nsight Systems GUI locally.

## What To Record In Your Notes

For each trace, write down:

- the exact server launch command
- whether CUDA graphs were enabled
- whether the forced sync was enabled
- the load generator command
- the trace filename
- the main visual difference you observed

This matters because later, when you choose an optimization target, you want to be able to explain why the trace suggests it.

## Cleanup

After the sync experiment:

1. remove or revert the temporary sync injection
2. relaunch the server in the normal configuration
3. keep the two `.nsys-rep` files

Do not leave the forced sync in your normal benchmark path.

## References

- Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/>
- Local `vllm` nsys helper README: `/Users/brandonaraki/projects/gpu-assignment/vllm/tools/profiler/nsys_profile_tools/README.md`
