# PLE GELU x Mul Fusion Experimental Protocol

## Goal

Evaluate `ple-gelu-and-mul-fusion` for `google/gemma-4-E2B-it` in the same compiled serving mode used for real decisions.

The key rule is unchanged from the decoder residual fusion protocol:

- **compiled / non-eager results are authoritative**
- eager-only results are diagnostic only

## Hypothesis

The mode:

```bash
--gemma4-kernel-experiment ple-gelu-and-mul-fusion
```

may reduce the cost of the Gemma4 PLE activation step:

1. `gate = per_layer_input_gate(hidden_states)`
2. `gelu(gate, approximate="tanh")`
3. `gated_per_layer = gate * per_layer_input`

But the experiment should advance only if that change survives compiled serving.

## Decision Rule

Use this order of evidence:

1. compiled AIPerf throughput / latency
2. compiled steady-state `nsys`
3. microbenchmark results for explanation only
4. eager-only profiling only if the compiled result needs explanation

Do not advance the experiment based only on microbenchmark or eager wins.

## Required Matrix

### Required Decision Sweep

- `baseline-compiled`
- `ple-gelu-and-mul-fusion-compiled`

These two runs should differ only in:

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment ple-gelu-and-mul-fusion
```

### Required Diagnostic Sweep

- `baseline-compiled-nsys`
- `ple-gelu-and-mul-fusion-compiled-nsys`
- `ple-gelu-and-mul-fusion-microbenchmark`

### Optional Secondary Diagnostics

- `baseline-eager`
- `ple-gelu-and-mul-fusion-eager`

Run these only if the compiled behavior needs explanation.

## Constants

Keep these fixed across the decision sweep:

- model: `google/gemma-4-E2B-it`
- `--tensor-parallel-size 1`
- `--dtype bfloat16`
- `--language-model-only`
- `--max-model-len 4096`
- `--max-num-seqs 8`
- `--max-num-batched-tokens 1024`
- `--gpu-memory-utilization 0.80`
- `--enable-chunked-prefill`
- `--async-scheduling`
- compilation config: `{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}`
- AIPerf synthetic input tokens: `512`
- AIPerf output tokens: `128`
- concurrency sweep: `1 2 4 8`

## Cache Hygiene

Before every server relaunch for a different mode:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

If any C++/CUDA source changed, rebuild first:

```bash
cd ~/vllm
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

## Artifact Layout

This protocol assumes:

- compiled baseline AIPerf:
  `~/gpu-assignment-results/step6-baseline`
- compiled fusion AIPerf:
  `~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion`
- protocol traces and logs:
  `~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol`
- microbenchmarks:
  `~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion/microbenchmarks`

## Commands

### 1. Start The Baseline Server

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh baseline
```

### 2. Run The Baseline AIPerf Sweep

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh baseline
```

### 3. Start The Fusion Server

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ple-gelu-and-mul-fusion
```

### 4. Run The Fusion AIPerf Sweep

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh ple-gelu-and-mul-fusion
```

### 5. Collect Steady-State `nsys`

Use two shells:

1. shell 1 runs the server and leaves it up
2. shell 2 runs the `nsys` wrapper below

The wrapper already includes a warm-up period before tracing starts. It sends the benchmark load immediately, waits `200` seconds, and records only the following `30` seconds. That warm-up window is there to exclude one-time work such as `torch.compile`, CUDA graph capture, allocator growth, and first-use kernel setup.

So the practical rule is:

- do **not** start `nsys` until the server is healthy
- do **not** start a second manual load generator unless you intentionally want extra pre-warming
- the script's own `--delay 200` is the normal warm-up mechanism

Baseline example:

Shell 1:

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh baseline
```

Wait until the server is ready to accept requests.

Shell 2:

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh baseline
```

Then repeat the same two-shell process for fusion.

Shell 1:

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ple-gelu-and-mul-fusion
```

Shell 2:

```bash
cd ~/gpu-assignment
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh ple-gelu-and-mul-fusion
```

If you want to shorten or lengthen the pre-trace warm-up window, override:

```bash
cd ~/gpu-assignment
WARMUP_SECONDS=120 CAPTURE_SECONDS=30 \
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh baseline
```

If you want the exact command that the wrapper runs, the baseline equivalent is:

```bash
cd ~/vllm
source .venv/bin/activate
PYTHONPATH=~/vllm nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --delay 200 \
  --duration 30 \
  --output ~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol/nsys/baseline_compiled \
  python -m vllm.benchmarks.serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://localhost:8000 \
    --model google/gemma-4-E2B-it \
    --dataset-name synthetic \
    --num-prompts 128 \
    --random-input-len 512 \
    --random-output-len 128 \
    --ignore-eos \
    --save-result \
    --result-dir ~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol/load
```

### 6. Run The Microbenchmark

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_microbenchmarks.sh
```

## Success Criteria

Advance the experiment only if all of the following hold:

- no correctness regression
- same API / server behavior
- measurable compiled throughput and/or latency improvement, or at minimum no regression with cleaner PLE scope traces
- `nsys` shows a cleaner PLE activation region under the new scope names
- no loss of CUDA graph compatibility
- no loss of async scheduling compatibility

If the microbenchmark wins but compiled AIPerf loses, the compiled result wins.
