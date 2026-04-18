# Part 1 Benchmarking

This document sets up the first benchmarking run for the assignment.

The goal of part 1 is narrow:

1. run vanilla `vllm` with `google/gemma-4-E2B-it`
2. keep the server configuration fixed at concurrency `4`
3. collect one `nsys` trace from the server itself
4. drive exactly one round of `4` concurrent requests during the capture window
5. save the commands, artifact locations, and a place to record results

This part does not compare kernel experiments. It is the baseline setup on unmodified `vllm`.

## Target Setup

- model: `google/gemma-4-E2B-it`
- server: vanilla `vllm serve`
- GPU count: `1`
- server-side concurrency cap: `--max-num-seqs 4`
- client load shape: `1` round of `4` simultaneous requests
- profiler: `nsys`

## Files Added For This Part

The script set for this benchmark lives in [gpu-assignment/scripts/part1_benchmarking](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking):

- [serve_vanilla_gemma4.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/serve_vanilla_gemma4.sh): runs the vanilla server without `nsys`
- [profile_vanilla_gemma4_nsys.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh): launches the server under `nsys`
- [run_single_round_load.sh](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/run_single_round_load.sh): drives the single concurrency-4 request round
- [single_round_chat_load.py](/Users/brandonaraki/projects/gpu-assignment/gpu-assignment/scripts/part1_benchmarking/single_round_chat_load.py): exact client implementation

## Why This Setup

The important detail is that `nsys` must wrap the server process, not the client process. The GPU work you care about happens inside `vllm serve`, so the profiler has to be attached there.

The client is deliberately simple:

- it waits for `http://localhost:8000/v1/models`
- it sends exactly `4` chat requests once
- it writes one JSON artifact with request latency and raw responses

That keeps the trace focused on one controlled burst of work instead of a long steady-state benchmark sweep.

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

- `nsys/vanilla_gemma4_e2b_c4_single_round.nsys-rep`
- `nsys/vanilla_gemma4_e2b_c4_single_round.qdrep` if your local `nsys` version emits it
- `load/vanilla_gemma4_e2b_c4_single_round.json`

## Optional Sanity Check Without Profiling

Use this if you want to confirm the server comes up before running `nsys`:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/serve_vanilla_gemma4.sh
```

In another shell:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/run_single_round_load.sh
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
- waits `90` seconds before capture
- records `20` seconds of trace time

Those defaults give the server time to finish startup, compilation, and graph capture before the profiler starts recording.

### Shell 2: Send The Single Request Round

After the server is healthy, and before the `nsys` delay window expires, run:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/part1_benchmarking/run_single_round_load.sh
```

Default load behavior:

- concurrency: `4`
- request rounds: `1`
- `max_tokens`: `128`
- endpoint: `/v1/chat/completions`
- output artifact: `~/gpu-assignment-results/part1-benchmarking/load/vanilla_gemma4_e2b_c4_single_round.json`

## Useful Overrides

Shorter warm-up:

```bash
cd ~/gpu-assignment/gpu-assignment
WARMUP_SECONDS=45 CAPTURE_SECONDS=15 \
scripts/part1_benchmarking/profile_vanilla_gemma4_nsys.sh
```

Different port:

```bash
cd ~/gpu-assignment/gpu-assignment
PORT=8001 BASE_URL=http://localhost:8001 \
scripts/part1_benchmarking/run_single_round_load.sh
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
scripts/part1_benchmarking/run_single_round_load.sh
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
- rounds:
- `max_tokens`:
- prompts used:

### Artifact Paths

- `nsys` trace:
- load JSON:
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
3. decide whether the single-round burst is enough or whether a second run should use a slightly longer or repeated burst
