# Async Output Sync Reduction Notes

## Goal

Evaluate a serving-path optimization for `Gemma 4 E2B` in `vLLM` that reduces
steady-state `cudaEventSynchronize` overhead in async scheduling by removing the
duplicate host wait on sampled-token device-to-host copies.

The important framing is:

- this is **not** a model-kernel fusion experiment
- this is a **serving-path** optimization in the v1 GPU worker
- the target is the async sampled-token handoff used by streaming decode
- the expected win is modest but defensible, especially at lower concurrency

## Why This Experiment Exists

The `nsys` baseline showed a large amount of time in `cudaEventSynchronize`
during steady-state decode. Repo inspection points to the async scheduling
output path as the most actionable source:

- `AsyncGPUModelRunnerOutput.get_output()` waits for the sampled-token D2H copy
  to finish before converting the copied tensor into Python lists
- `GPUInputBatch.update_async_output_token_ids()` can wait on the same copy a
  second time when logits processors need repaired `output_token_ids`

That means one decode step can pay for the same sampled-token copy twice on the
CPU side. This experiment replaces the raw `(cpu_tensor, event)` handoff with a
shared sampled-token result object so both consumers reuse the same readiness
state and `tolist()` materialization.

## Public Experiment Mode

The experiment is exposed behind the existing Gemma 4 step-6 flag:

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment async-output-sync-reduction
```

Mode semantics:

- `baseline`: current async output-copy behavior
- `async-output-sync-reduction`: enables the shared sampled-token handoff so
  the async output path synchronizes and materializes sampled token IDs at most
  once per decode step

No extra compile overrides are required beyond the standard step-6 launch
configuration.

## Experiment Design

This experiment is intentionally narrow:

- same model
- same async scheduling
- same CUDA graph / compile setup
- same AIPerf sweep shape
- only the `--gemma4-kernel-experiment` value changes

The hypothesis is:

- compiled step-6 throughput is being limited in part by duplicate host waits
- removing the duplicate sampled-token wait should reduce
  `cudaEventSynchronize` count in compiled traces
- throughput improvements should show up most clearly for streaming decode at
  lower concurrency (`c1`, `c2`)

## Recorded Findings

The first compiled `nsys` traces for this experiment included server startup,
compile, and graph-capture activity, so they were not reliable for attributing
steady-state bottlenecks. After rerunning `nsys` with delayed capture to isolate
the serving window, the conclusion became much clearer:

- the async output sync change worked mechanically
- it reduced `cudaEventSynchronize`
- but `cudaEventSynchronize` was already a tiny steady-state cost
- so the experiment had almost no effect on end-to-end step-6 performance

### Steady-State `nsys` Results

Steady-state compiled traces showed:

- baseline `cudaEventSynchronize`: `20.5 ms` total across `2732` calls
- async-output-sync-reduction `cudaEventSynchronize`: `16.5 ms` total across
  `2278` calls

So the optimization reduced sync overhead, but only by a few milliseconds over
the entire captured window. That is far too small to drive a meaningful
throughput improvement.

The larger steady-state signals were:

- `preprocess` remained the largest meaningful NVTX bucket
  - baseline median: `1.704 ms`
  - async-output-sync-reduction median: `1.677 ms`
- `sample` remained the next largest steady-state region
  - baseline median: `0.914 ms`
  - async-output-sync-reduction median: `0.894 ms`
- `forward` median remained much smaller
  - baseline median: `0.144 ms`
  - async-output-sync-reduction median: `0.145 ms`

The CUDA API picture in steady state also changed relative to the contaminated
startup traces:

- `cudaLaunchKernel` was the largest CUDA API bucket
  - baseline: `1.185 s`
  - async-output-sync-reduction: `0.952 s`
- `cudaMemcpyAsync` was second
  - baseline: `0.471 s`
  - async-output-sync-reduction: `0.387 s`
- host-to-device transfer volume was tiny in absolute terms
  - baseline: `8.657 MB`
  - async-output-sync-reduction: `7.218 MB`

This means the steady-state serving path is dominated much more by many small
kernel launches and model compute than by large host-to-device payloads or
output-side synchronization.

### What This Experiment Proved

This experiment successfully validated the narrow diagnosis it was built to
test:

- the async sampled-token handoff did perform duplicate waits in the baseline
- the shared sampled-token result object removed that duplication
- both consumers now reuse the same readiness/materialization state

But it also falsified the stronger step-6 performance hypothesis:

- duplicate sampled-token waits were not a meaningful end-to-end throughput
  limiter in steady-state compiled serving

### Practical Conclusion

For step 6, async output sync reduction should be treated as a completed
diagnostic experiment rather than an optimization direction worth extending:

- it improved a real but tiny overhead
- it did not materially change the critical path
- more work in this area is unlikely to pay off

The next low-hanging fruit should come from either:

- reducing preprocess-time launch count / per-step metadata work, or
- improving model compute in the dominant decoder path

The strongest steady-state GPU signal is still the main `gemvx` kernel family,
which accounts for about `79.5%` of GPU kernel time in both baseline and
experiment traces. That makes compute-path optimization a better next target
than more async output handoff cleanup.

## Commands To Run The Experiment

This section assumes:

- `vllm` and `AIPerf` run on the EC2 instance
- the model is `google/gemma-4-E2B-it`
- baseline benchmark artifacts live under `~/gpu-assignment-results/step6-baseline`
- experiment artifacts live under `~/gpu-assignment-results/step6-async-output-sync-reduction`
- plotting is run from the cloned `gpu-assignment` repo at `~/gpu-assignment`

### 1. Start The Baseline Server

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

### 2. Run The Baseline AIPerf Sweep

```bash
mkdir -p ~/gpu-assignment-results/step6-baseline
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Baseline concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-baseline/baseline_c${C}
done
```

### 3. Start The Async Output Sync Reduction Server

Stop the baseline server, then restart:

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
  --gemma4-kernel-experiment async-output-sync-reduction
```

### 4. Run The Async Output Sync Reduction Sweep

```bash
mkdir -p ~/gpu-assignment-results/step6-async-output-sync-reduction
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Async output sync reduction concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-async-output-sync-reduction/async_output_sync_reduction_c${C}
done
```

### 5. Generate A Combined Pareto Plot

```bash
cd ~/gpu-assignment

mkdir -p ~/gpu-assignment-results/step6-comparison

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results \
  --experiment baseline \
  --experiment async-output-sync-reduction \
  --output-csv ~/gpu-assignment-results/step6-comparison/async_output_sync_reduction_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/async_output_sync_reduction_pareto.png \
  --title 'Gemma 4 E2B Async Output Sync Reduction Comparison'
```

Optional broader comparison if other step-6 artifacts already exist:

```bash
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results \
  --experiment baseline \
  --experiment async-output-sync-reduction \
  --experiment decoder-residual-fusion \
  --experiment qk-norm-rope-fusion-lt-512 \
  --experiment qk-norm-rope-fusion-512 \
  --output-csv ~/gpu-assignment-results/step6-comparison/combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/combined_pareto.png \
  --title 'Gemma 4 E2B Step-6 Experiment Comparison'
```

## Success Criteria

The original success criteria were:

- no correctness regression in streaming outputs
- compiled `nsys` traces show materially fewer `cudaEventSynchronize` calls per
  `forward`
- AIPerf output tokens/sec does not regress versus baseline
- the preferred outcome is an improvement at `c1` or `c2`

Observed outcome after rerunning `nsys` in a delayed steady-state window:

- correctness was preserved
- compiled `nsys` traces showed a small but real drop in
  `cudaEventSynchronize`
- throughput improvement was negligible

So the experiment validated the diagnosis about duplicate waits, but falsified
the stronger hypothesis that those waits were a meaningful steady-state
throughput limiter.

## Caveats

- Step-5 eager `nsys` traces are useful for diagnosis, but final step-6 claims
  should be based on the normal compiled / CUDA-graph serving config.
- This experiment does not change the dominant model compute kernels, so expect
  incremental rather than dramatic wins.
- If throughput regresses while `cudaEventSynchronize` drops, the regression is
  likely elsewhere in the serving path and should be investigated separately.
- The initial non-delayed compiled traces were startup-contaminated and should
  not be used to attribute steady-state bottlenecks.
- In the delayed steady-state traces, the main signals were preprocess, sample,
  kernel-launch overhead, and dominant decoder compute rather than output-side
  sync.
