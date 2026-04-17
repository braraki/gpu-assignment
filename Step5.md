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

## High-Level Plan

Run step 5 in this order:

1. collect a baseline `.nsys-rep`
2. collect a second `.nsys-rep` with a forced sync point
3. run `nsys stats` for quick summaries
4. optionally run the local `vllm` helper in `tools/profiler/nsys_profile_tools`
5. download the `.nsys-rep` files and inspect them in the Nsight Systems GUI on your laptop

## Baseline Trace

Stop any already-running server first, then relaunch `vllm` under `nsys`.

On the EC2 instance:

```bash
cd ~/vllm
source .venv/bin/activate

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --delay 240 \
  --duration 45 \
  --output ~/gpu-assignment-results/step5-nsys/baseline \
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
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Why these `nsys` flags:

- `--delay 20` gives the server time to start before capture begins
- `--duration 45` keeps the trace short enough to be practical
- `--trace=cuda,nvtx,osrt` captures the CUDA path plus runtime markers
- `--cuda-graph-trace=node` is useful for the graph-enabled baseline
- `--trace-fork-before-exec=true` matches the local `vllm` profiler guidance

### Generate Load During The Capture Window

In a second SSH session, after starting the command above, run a short AIPerf load:

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
  --concurrency 4 \
  --request-count 256 \
  --warmup-request-count 8 \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step5-nsys/baseline_load
```

When `nsys` exits, you should have:

```text
~/gpu-assignment-results/step5-nsys/baseline.nsys-rep
```

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

VLLM_FORCE_SYNC_POINT=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --delay 20 \
  --duration 45 \
  --output ~/gpu-assignment-results/step5-nsys/forced_sync \
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
  --concurrency 4 \
  --request-count 256 \
  --warmup-request-count 8 \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
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
