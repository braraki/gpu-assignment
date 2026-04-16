# Step 4: Benchmark The Running Server With AIPerf

This document walks through step 4 of the assignment:

1. keep the `vllm` server running with the maximum concurrency you want to test
2. use `AIPerf` to benchmark different client-side concurrencies against that same server
3. collect one data point per concurrency
4. build the Pareto curve:
   - x-axis: token/s/user
   - y-axis: token/s/gpu

This version is written for the current setup:

- model: `google/gemma-4-E2B-it`
- server already running on the EC2 instance
- single GPU: `g6.xlarge`
- current `Step3` server config caps practical concurrency at `8`
- run `AIPerf` on the EC2 instance itself, not from your laptop, so WAN/SSH-tunnel latency does not pollute the numbers

## Why Use AIPerf Here

The assignment specifically mentions `AIPerf`, and it is a good fit for this step.

Why:

- `AIPerf` supports OpenAI-compatible text APIs.
- It has a built-in concurrency benchmarking mode.
- It writes structured artifacts for each run, so you are not forced to rely only on terminal copy/paste.
- It directly reports:
  - `Output Token Throughput Per User (tokens/sec/user)`
  - `Output Token Throughput (tokens/sec)`

Those line up cleanly with the assignment’s Pareto curve:

- x-axis: `token/s/user`
- y-axis: `tokens/s/gpu`

Because you are using one GPU, `tokens/s/gpu` is the same as total output token throughput.

## What We Are Measuring

For each concurrency value, record:

- `Output Token Throughput Per User`
- `Output Token Throughput`
- `Time to First Token`
- `Inter Token Latency`
- `Request Latency`

For the Pareto curve:

- x-axis = `Output Token Throughput Per User`
- y-axis = `Output Token Throughput / gpu_count`

Since `gpu_count = 1` here:

- y-axis = `Output Token Throughput`

## Benchmark Strategy

The assignment says:

- start the server with the maximum concurrency
- benchmark different concurrencies against the same server

For the current setup:

- the server is already configured with `--max-num-seqs 8`
- so the first sweep should use:
  - `1`
  - `2`
  - `4`
  - `8`

That gives you four initial Pareto points without changing the server configuration between runs.

## Before You Start

Confirm the server is still alive:

```bash
curl http://localhost:8000/v1/models
```

If that works, continue.

If it fails, restart the server using `Step3.md`.

## Install AIPerf

Install `AIPerf` in a separate virtual environment so you do not disturb the working `vllm` environment.

On the EC2 instance:

```bash
python3 -m venv ~/aiperf-venv
source ~/aiperf-venv/bin/activate
pip install --upgrade pip
pip install aiperf
```

Quick sanity check:

```bash
aiperf --help
```

## Benchmark Parameters

Use these benchmark settings for the first pass:

- endpoint type: `chat`
- streaming: enabled
- UI mode: `simple`
- tokenizer: `google/gemma-4-E2B-it`
- request count: `128`
- warmup request count: `8`
- synthetic input tokens mean: `512`
- synthetic input tokens stddev: `0`
- output tokens mean: `128`
- output tokens stddev: `0`
- random seed: `0`

Why these values:

- fixed input and output targets make concurrency points easier to compare
- `128` requests is enough to get a stable first result without making each run too slow
- a small warmup helps reduce cold-start distortion

## Important Output-Length Note

`AIPerf` documents that output length targets are not always guaranteed unless you also pass extra generation controls such as `ignore_eos` and, if supported by the server, min/max token controls.

For the first pass, use:

```text
--extra-inputs ignore_eos:true
```

If you later need tighter control over exact output length, the AIPerf docs recommend also trying extra inputs such as `max_tokens` and `min_tokens` if your server accepts them. Do not block the first benchmark sweep on that.

## Create A Result Directory

```bash
mkdir -p ~/gpu-assignment-results/step4-aiperf
```

## Run One Benchmark Manually

Start with concurrency `1`:

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
  --request-count 128 \
  --warmup-request-count 8 \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step4-aiperf/baseline_c1
```

If successful, `AIPerf` should print a metrics table that includes:

- `Time to First Token (ms)`
- `Inter Token Latency (ms)`
- `Request Latency (ms)`
- `Output Token Throughput Per User`
- `Output Token Throughput`

That is already enough to produce one Pareto point.

## Run The Full Concurrency Sweep

Then run the sweep:

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Running concurrency ${C} ==="
  aiperf profile \
    --model google/gemma-4-E2B-it \
    --tokenizer google/gemma-4-E2B-it \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --endpoint v1/chat/completions \
    --streaming \
    --ui simple \
    --concurrency "${C}" \
    --request-count 128 \
    --warmup-request-count 8 \
    --synthetic-input-tokens-mean 512 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --random-seed 0 \
    --artifact-dir ~/gpu-assignment-results/step4-aiperf/baseline_c${C}
done
```

This should create one artifact directory per concurrency value.

## What To Save

For each run, keep at least:

- the terminal summary
- `profile_export_aiperf.json`
- `profile_export_aiperf.csv`
- the log file

Depending on the exact version, `AIPerf` may also emit additional raw files such as per-request exports or generated inputs. Keep those too if they are present.

## Build The Pareto Table

Do not build the table by hand. Use the saved `AIPerf` JSON artifacts as the source of truth.

Because this is a single-GPU run:

- `tokens_per_s_per_gpu = Output Token Throughput`

In this repo, name the saved runs:

- `baseline_c1`
- `baseline_c2`
- `baseline_c4`
- `baseline_c8`

## Plot The Curve From Saved Results

This repo includes a helper script that gathers all `baseline_c*` runs, writes a summary CSV, and saves the Pareto curve to disk. If you cloned this repo onto the EC2 instance, install the plotting dependency first:

```bash
cd ~/gpu-assignment
python3 -m pip install -r requirements.txt
```

Run it from this repo:

```bash
cd ~/gpu-assignment
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step4-aiperf \
  --output-csv ~/gpu-assignment-results/step4-aiperf/baseline_summary.csv \
  --output-figure ~/gpu-assignment-results/step4-aiperf/baseline_pareto.png
```

Expected input layout:

- `~/gpu-assignment-results/step4-aiperf/baseline_c1/profile_export_aiperf.json`
- `~/gpu-assignment-results/step4-aiperf/baseline_c2/profile_export_aiperf.json`
- `~/gpu-assignment-results/step4-aiperf/baseline_c4/profile_export_aiperf.json`
- `~/gpu-assignment-results/step4-aiperf/baseline_c8/profile_export_aiperf.json`

Outputs:

- `~/gpu-assignment-results/step4-aiperf/baseline_summary.csv`
- `~/gpu-assignment-results/step4-aiperf/baseline_pareto.png`

The summary CSV contains one row per run and includes:

- concurrency
- `Output Token Throughput Per User`
- `tokens/s/gpu`
- `Output Token Throughput`
- `Time to First Token`
- `Inter Token Latency`
- `Request Latency`

For the assignment, the most important two remain:

- `Output Token Throughput Per User`
- `Output Token Throughput`

## How To Interpret The Curve

Typical pattern:

- low concurrency:
  - better user experience
  - better tokens/sec/user
  - lower total throughput
- higher concurrency:
  - worse tokens/sec/user
  - higher latency
  - better total throughput

The Pareto curve is the tradeoff between:

- user-level speed
- overall GPU efficiency

## If AIPerf Fails

Common checks:

1. If the server is unreachable:
   - verify `curl http://localhost:8000/v1/models`
2. If the endpoint errors:
   - verify the server is serving `/v1/chat/completions`
3. If the output length is much shorter than requested:
   - this can happen even with `--output-tokens-mean`
   - keep the result for now
   - add a note in the CSV
4. If concurrency `8` is unstable:
   - keep the successful `1, 2, 4` runs
   - note that `8` exceeded the stable operating point

## Why This Matches The Assignment

This workflow follows the assignment literally:

- the `vllm` server stays up with a fixed maximum concurrency configuration
- `AIPerf` is the benchmark client
- you sweep client-side concurrency values one after another against the same server
- each run produces one Pareto point:
  - x-axis: `Output Token Throughput Per User`
  - y-axis: `Output Token Throughput / gpu_count`

## Optional Fallback: vLLM Bench Serve

If `AIPerf` turns out to be blocked by packaging or CLI issues on the instance, you can still use `vllm bench serve` as a fallback benchmark client.

However, for this assignment step, `AIPerf` should be treated as the primary path because:

- it is explicitly mentioned in the assignment
- it directly reports the per-user throughput metric you need

## References

- AIPerf README: <https://github.com/ai-dynamo/aiperf>
- NVIDIA NIM guide using AIPerf: <https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/step-by-step.html>
- Current local `vllm` benchmark implementation: `/Users/brandonaraki/projects/gpu-assignment/vllm/vllm/benchmarks/serve.py`
