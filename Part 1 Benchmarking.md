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
- concurrency control: `4`

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
- waits `140` seconds before capture
- records `30` seconds of trace time

Those defaults assume the server takes about `80` seconds to become ready and leave about `60` additional seconds for post-startup warm-up before the profiler starts recording.

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

### Environment

- machine:
- GPU:
- CUDA driver:
- `vllm` commit:
- `nsys` version:
- date:

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

## Next Step

After the first trace is saved, the next useful follow-up is:

1. run `nsys stats` on the saved `.nsys-rep`
2. inspect the trace in the Nsight Systems GUI
3. decide whether the first `c=4` load window is enough or whether a second run should use a longer capture window or a larger `request_count`
