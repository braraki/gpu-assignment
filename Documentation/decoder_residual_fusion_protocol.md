# Decoder Residual Fusion Experimental Protocol

## Goal

Evaluate `decoder-residual-fusion` for `google/gemma-4-E2B-it` in the same
execution mode used for real serving decisions.

The key rule for this protocol is:

- **compiled / non-eager results are authoritative**
- eager results are allowed only as secondary diagnostics
- if eager and compiled disagree, the compiled result wins

This protocol exists because the fused residual path can look good in eager
mode while regressing in compiled mode due to different graph lowering,
functionalization, custom-op dispatch, or CUDA graph behavior.

## Hypothesis

The experiment mode:

```bash
--gemma4-kernel-experiment decoder-residual-fusion
```

may reduce the cost of the decoder transition:

1. `post_attention_layernorm`
2. residual add
3. `pre_feedforward_layernorm`

But that hypothesis must be tested in compiled serving, not inferred from eager
microbenchmarks or eager `nsys`.

## Decision Rule

Use this order of evidence:

1. compiled AIPerf throughput / latency
2. compiled steady-state `nsys`
3. compiler diagnostics (`TORCH_LOGS`)
4. eager-only profiling, only if the compiled result needs explanation

Do **not** advance this optimization based on eager-only wins.

## Experiment Matrix

Run the following configurations in this order.

### Required Decision Sweep

- `baseline-compiled`
- `decoder-residual-fusion-compiled`

These two runs should differ only in:

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment decoder-residual-fusion
```

### Required Diagnostic Sweep

- `baseline-compiled-nsys`
- `decoder-residual-fusion-compiled-nsys`
- `baseline-compiled-torch-logs`
- `decoder-residual-fusion-compiled-torch-logs`

### Optional Sensitivity Sweep

Run this only if the fused mode loses in compiled serving and you need to know
whether the issue is tied to custom-op dispatch:

- `decoder-residual-fusion-compiled-custom-op-rms`

That run explicitly enables the `rms_norm` custom op in the compilation config.

### Optional Secondary Diagnostic

- `baseline-eager`
- `decoder-residual-fusion-eager`

These runs are only for explanation and should not drive the decision.

## Constants

Keep these fixed across all decision runs:

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
- endpoint: `/v1/chat/completions`
- streaming enabled
- AIPerf synthetic input tokens: `512`
- AIPerf output tokens: `128`
- concurrency sweep: `1 2 4 8`
- same machine, same GPU, same software environment

## Cache Hygiene

Compiled experiments are invalid if they reuse a stale torch compile cache after
changing:

- `--gemma4-kernel-experiment`
- compilation config
- any relevant `vllm` source

Before every server relaunch for a different experiment mode, clear:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

If any C++ or CUDA extension source changed, rebuild `vllm` before rerunning:

```bash
cd ~/vllm
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

## Artifact Layout

This protocol assumes results are written under:

- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/baseline`
- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion`
- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion_rms_custom_op`
- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/eager`
- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/nsys`
- `~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/logs`

## 1. Baseline Compiled Throughput Sweep

### Start The Baseline Server

```bash
cd ~/vllm
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES=0 \
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

### Run The Baseline AIPerf Sweep

```bash
mkdir -p ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/baseline
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Baseline compiled concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/baseline/c${C}
done
```

## 2. Fusion Compiled Throughput Sweep

### Start The Fusion Server

Stop the baseline server, clear the compile cache, then restart:

```bash
cd ~/vllm
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES=0 \
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

### Run The Fusion AIPerf Sweep

```bash
mkdir -p ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Fusion compiled concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion/c${C}
done
```

## 3. Compiled Steady-State `nsys`

Do not profile server startup, graph capture, or initial compile. Use delayed
capture so the trace reflects steady-state serving only.

### Recommended Method

1. start the server normally
2. send warmup traffic until recompiles and graph capture settle
3. attach `nsys` or use delayed profiling for the steady-state window
4. run the same request shape used by AIPerf

### Required Settings

Enable NVTX scopes for readability:

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

Collect at least:

- NVTX summary
- CUDA API summary
- CUDA GPU kernel summary

### Baseline Compiled `nsys`

Write the final exported text under:

```text
~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/nsys/baseline_compiled.txt
```

### Fusion Compiled `nsys`

Write the final exported text under:

```text
~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/nsys/fusion_compiled.txt
```

### What To Compare

Check whether fusion changes:

- `:gpu_model_runner: forward`
- `:gpu_model_runner: sample`
- `cudaEventSynchronize`
- kernel launch count
- `fused_add_rms_norm` or related RMSNorm kernels
- recompilation or graph-capture activity leaking into the trace

If the fused mode loses in AIPerf and also loses in steady-state compiled
`nsys`, treat that as a real regression.

## 4. Compiler Diagnostics

Run one server launch per mode with compiler logging enabled and save the logs.

### Baseline Compiler Diagnostics

```bash
cd ~/vllm
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache
mkdir -p ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/logs

TORCH_LOGS=recompiles,graph_breaks \
CUDA_VISIBLE_DEVICES=0 \
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
  --gemma4-kernel-experiment baseline \
  2>&1 | tee ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/logs/baseline_torch_logs.txt
```

### Fusion Compiler Diagnostics

```bash
cd ~/vllm
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

TORCH_LOGS=recompiles,graph_breaks \
CUDA_VISIBLE_DEVICES=0 \
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
  --gemma4-kernel-experiment decoder-residual-fusion \
  2>&1 | tee ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/logs/fusion_torch_logs.txt
```

### What To Look For

- extra graph breaks in fusion mode
- extra recompiles in fusion mode
- shape specialization differences
- evidence that the fused path changes graph partitioning or capture behavior

## 5. Optional Custom-Op Sensitivity Check

Use this only if compiled fusion loses and you need to test whether the loss is
connected to the `rms_norm` custom-op path.

### Start The Fusion Server With Explicit `+rms_norm`

```bash
cd ~/vllm
source .venv/bin/activate

rm -rf ~/.cache/vllm/torch_compile_cache

CUDA_VISIBLE_DEVICES=0 \
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
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["none","+rms_norm"]}' \
  --gemma4-kernel-experiment decoder-residual-fusion
```

### Run The Diagnostic Sweep

```bash
mkdir -p ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion_rms_custom_op
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Fusion compiled +rms_norm concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/fusion_rms_custom_op/c${C}
done
```

### Interpretation

- if `+rms_norm` materially recovers the regression, the issue is likely tied to
  compiled lowering of the native residual-aware RMSNorm path
- if it does not recover, the regression is more likely structural to the fused
  experiment mode or its effect on graph optimization

## 6. Optional Eager Diagnostic

Only run this after the compiled conclusion is already known.

The purpose is:

- to explain why compiled and eager disagree
- not to choose the optimization direction

If you run eager, use the same request shape and artifact layout, but label the
results clearly as non-decision data.

## 7. Comparison And Reporting

Generate a combined comparison after the required sweeps finish.

```bash
cd ~/gpu-assignment
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol \
  --experiment baseline \
  --experiment fusion \
  --experiment fusion_rms_custom_op \
  --output-csv ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-decoder-residual-fusion-protocol/combined_pareto.png \
  --title 'Decoder Residual Fusion Protocol Comparison'
```

If the plotting helper expects the original step-6 naming convention, adapt the
artifact directory names rather than changing the experimental logic.

## Acceptance Criteria

Advance `decoder-residual-fusion` only if all of the following hold:

- compiled AIPerf improves at one or more meaningful operating points
- compiled AIPerf does not show a clear regression at the other key points
- steady-state compiled `nsys` shows a plausible mechanism for the win
- compiler diagnostics do not show materially worse recompilation or graph-break
  behavior

Reject or deprioritize it if:

- eager wins but compiled loses
- compiled throughput is flat and latency worsens
- the fused mode creates unstable compile behavior
- the apparent win only appears in startup-contaminated traces

## Minimum Report Template

For each run, record:

- commit SHA
- launch command
- compilation config
- whether compile cache was cleared
- AIPerf outputs for `c1`, `c2`, `c4`, `c8`
- steady-state `nsys` summary path
- compiler log path
- final conclusion: `advance`, `hold`, or `drop`

## Practical Recommendation

For this experiment, the shortest valid path is:

1. baseline compiled AIPerf
2. fusion compiled AIPerf
3. baseline compiled steady-state `nsys`
4. fusion compiled steady-state `nsys`
5. compiler logs for both modes if fusion loses

That is enough to decide whether decoder residual fusion is still worth pursuing.
