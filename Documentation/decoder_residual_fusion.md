# Decoder Residual Fusion Notes

## Goal

Improve `Gemma 4 E2B` serving efficiency in `vLLM`, with the first step-6 PR focused on a low-risk decoder kernel enhancement that remains compatible with:

- CUDA graphs
- async scheduling
- AIPerf benchmarking

The immediate target is **decoder residual fusion**, specifically reusing vLLM's existing fused add + RMSNorm path rather than starting with a brand-new Triton kernel.

## Why E2B Is A Dense-Model Optimization Problem

`Gemma 4 E2B` is the wrong place to start with MoE kernel work.

- The practical hot path for this assignment is dense decoder inference.
- The current `gemma4.py` path still has unfused residual-add + norm structure in the decoder.
- That makes residual fusion a better first target than router / expert kernels.

## Ranked Target List

1. Decoder residual fusion
2. PLE branch fusion
3. Q/K norm + RoPE fusion
4. Attention backend checks only if `nsys` traces show an unexpected fallback or bubble

MoE-specific kernel work is intentionally deferred for this model.

## Decoder Residual Fusion Hypothesis

The strongest first target is the transition:

1. `post_attention_layernorm`
2. residual add
3. `pre_feedforward_layernorm`

In the current Gemma 4 decoder flow, that sequence is still materialized explicitly. Comparable dense models in vLLM already use residual threading and fused add + RMSNorm semantics.

The experiment is:

- keep `baseline` as the current behavior
- add a gated `decoder-residual-fusion` mode
- in experiment mode, replace the explicit post-attention residual add followed by `pre_feedforward_layernorm(...)` with the existing fused `RMSNorm(x, residual)` path

This keeps the change narrow:

- no new custom kernel in the first pass
- no PLE fusion in the same PR
- no Q/K + RoPE fusion in the same PR

## Experiment Design

Two server launch modes must exist:

- `baseline`
- `decoder-residual-fusion`

The switch must be exposed as a dedicated `vllm serve` flag:

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment decoder-residual-fusion
```

That flag should be the only server-side difference when collecting AIPerf numbers.

## Commands To Run The Experiment

This section assumes:

- the `vllm` server and `AIPerf` are both run on the EC2 instance
- the model is `google/gemma-4-E2B-it`
- baseline benchmark artifacts are written under `~/gpu-assignment-results/step6-baseline`
- decoder residual fusion artifacts are written under `~/gpu-assignment-results/step6-decoder-residual-fusion`
- the plotting helper is run from the cloned `gpu-assignment` repo at `~/gpu-assignment`

### 1. Start The Baseline Server

```bash
cd ~/vllm
source .venv/bin/activate

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

### 3. Start The Decoder Residual Fusion Server

Stop the baseline server, then restart with the experiment flag changed:

```bash
cd ~/vllm
source .venv/bin/activate

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

### 4. Run The Decoder Residual Fusion AIPerf Sweep

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Decoder residual fusion concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-decoder-residual-fusion/decoder_residual_fusion_c${C}
done
```

### 5. Generate A Combined Pareto Plot

```bash
cd ~/gpu-assignment
python3 -m pip install -r requirements.txt

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results \
  --experiment baseline \
  --experiment decoder-residual-fusion \
  --experiment qk-norm-rope-fusion-lt-512 \
  --experiment qk-norm-rope-fusion-512 \
  --output-csv ~/gpu-assignment-results/step6-comparison/combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/combined_pareto.png \
  --title 'Gemma 4 E2B Kernel Experiment Comparison'
```

### 6. Compare The Result Sets

After the plot command finishes, the key artifacts should be:

- `combined_summary.csv`
- `combined_pareto.png`
- `combined_pareto_total_tokens.png`

`combined_pareto.png` plots output tokens per second, while `combined_pareto_total_tokens.png` plots total tokens per second. The combined CSV includes a `series_name` column so baseline, decoder residual fusion, LT-512 Q/K-norm+RoPE fusion, and CUDA 512 Q/K-norm+RoPE fusion points can still be filtered separately. Use those files together with the matching `nsys` traces when evaluating whether the experiment is promising.

## Profiling And Benchmark Artifacts

Collect the same artifacts for both modes.

### AIPerf

Use the same sweep as step 4, but store outputs separately:

- `baseline_c1`
- `baseline_c2`
- `baseline_c4`
- `baseline_c8`
- `decoder_residual_fusion_c1`
- `decoder_residual_fusion_c2`
- `decoder_residual_fusion_c4`
- `decoder_residual_fusion_c8`

The goal is to compare the Pareto curve with only the experiment flag changed.

### Nsight Systems

Collect matching traces for:

- baseline
- decoder residual fusion experiment

Use the same prompt shape, concurrency setup, and server config wherever possible.

Trace review should focus on:

- fewer small norm/add kernel boundaries around the decoder transition
- reduced bubble time in that region
- no obvious host syncs introduced by the experiment

## Success Criteria

The experiment is considered promising only if all of the following hold:

- no correctness regression
- same server and API behavior
- measurable throughput and/or latency improvement
- no loss of CUDA graph compatibility
- no loss of async scheduling compatibility

## Follow-On Targets After This PR

If decoder residual fusion is safe and measurable, the next candidates are:

1. fuse the PLE gate / activation / multiply path
2. fuse Q/K normalization with RoPE preparation
3. revisit attention backend behavior only if traces suggest the fast path is not being used
