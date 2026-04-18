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
scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh baseline
```

### 2. Run The Baseline AIPerf Sweep

```bash
scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh baseline
```

### 3. Start The Fusion Server

```bash
scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ple-gelu-and-mul-fusion
```

### 4. Run The Fusion AIPerf Sweep

```bash
scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh ple-gelu-and-mul-fusion
```

### 5. Collect Steady-State `nsys`

Use two shells:

1. shell 1 runs the server under `nsys`
2. shell 2 drives load into that server

The earlier PLE wrapper was incorrect because it ran `nsys` around the client-side benchmark process. That does not capture the server GPU work you actually care about. The corrected wrapper now runs `nsys` around `vllm serve`, which matches the intent of the older steady-state protocols.

The wrapper includes a delayed capture window. It launches the server under `nsys`, waits `200` seconds, and records only the following `30` seconds. That warm-up window is there to exclude one-time work such as `torch.compile`, CUDA graph capture, allocator growth, and first-use kernel setup.

So the practical rule is:

- shell 1 runs `run_nsys_protocol.sh`, not `serve_gemma4_experiment.sh`
- shell 2 must send traffic during the delay window so the captured 30-second slice contains steady-state decoding work
- the script's own `--delay 200` is the normal warm-up mechanism

Baseline example:

Shell 1:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh baseline
```

Wait until the server is ready to accept requests, then start shell 2.

Shell 2:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_load.sh baseline
```

Then repeat the same two-shell process for fusion.

Shell 1:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh ple-gelu-and-mul-fusion
```

Shell 2:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_load.sh ple-gelu-and-mul-fusion
```

If you want to shorten or lengthen the pre-trace warm-up window, override:

```bash
cd ~/gpu-assignment/gpu-assignment
WARMUP_SECONDS=120 CAPTURE_SECONDS=30 \
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh baseline
```

If you want the exact command that the wrapper runs, the baseline equivalent is:

```bash
cd ~/vllm
source .venv/bin/activate
rm -rf ~/.cache/vllm/torch_compile_cache
VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 \
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-graph-trace=node \
  --delay 200 \
  --duration 30 \
  --output ~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion-protocol/nsys/baseline_compiled \
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

The matching shell 2 load command is:

```bash
cd ~/gpu-assignment/gpu-assignment
scripts/step6_ple_gelu_and_mul_fusion/run_nsys_load.sh baseline
```

### 6. Run The Microbenchmark

```bash
scripts/step6_ple_gelu_and_mul_fusion/run_microbenchmarks.sh
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
