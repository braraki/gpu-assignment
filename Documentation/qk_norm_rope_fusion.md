# Q/K RMSNorm + RoPE Fusion Notes

## Goal

Evaluate a second `Gemma 4 E2B` kernel experiment in `vLLM` using the existing fused Q/K RMSNorm + RoPE infrastructure.

This experiment is intended to be:

- lower-risk than a new Gemma-4-only kernel
- compatible with CUDA graphs
- compatible with async scheduling
- easy to benchmark with the existing one-flag `AIPerf` workflow

## Why This Is The Next Experiment

This is the best follow-on to decoder residual fusion because the repository already contains:

- a fused `fused_qk_norm_rope` custom op
- a compile pass that matches the unfused Q/K RMSNorm + RoPE pattern
- kernel tests and compile-pass tests for that path

That makes this experiment mostly a **controlled enablement problem** rather than a brand-new kernel implementation.

## Why This Is Lower Risk Than PLE Fusion

PLE fusion is still interesting, but it would require new Gemma-4-specific fusion work around:

- `per_layer_input_gate`
- `gelu(..., approximate="tanh")`
- `gate * per_layer_input`
- `per_layer_projection`

By contrast, Q/K-norm+RoPE fusion already exists generically in `vLLM`, so this experiment can validate whether Gemma 4 E2B benefits materially from it before spending time on a more bespoke PLE path.

## Experiment Description

The experiment mode is:

- `qk-norm-rope-fusion`

The public interface remains the same single flag used for the other Gemma 4 experiments:

```bash
--gemma4-kernel-experiment qk-norm-rope-fusion
```

When this mode is selected, the server should automatically enable the compile-time machinery needed for the existing fusion path:

- `pass_config.enable_qk_norm_rope_fusion = True`
- `+rms_norm`
- `+rotary_embedding`

No manual `compilation-config` changes should be required beyond the existing step-3 launch command.

## Commands To Run The Experiment

This section assumes:

- the `vllm` server and `AIPerf` are both run on the EC2 instance
- the model is `google/gemma-4-E2B-it`
- benchmark artifacts are written under `~/gpu-assignment-results/step6-qk-norm-rope-fusion`
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
mkdir -p ~/gpu-assignment-results/step6-qk-norm-rope-fusion
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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion/baseline_c${C}
done
```

### 3. Start The Q/K-Norm + RoPE Fusion Server

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
  --gemma4-kernel-experiment qk-norm-rope-fusion
```

### 4. Run The Q/K-Norm + RoPE AIPerf Sweep

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Q/K-norm + RoPE fusion concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_c${C}
done
```

### 5. Generate The Baseline Pareto Plot

```bash
cd ~/gpu-assignment
python3 -m pip install -r requirements.txt

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step6-qk-norm-rope-fusion \
  --pattern 'baseline_c*' \
  --output-csv ~/gpu-assignment-results/step6-qk-norm-rope-fusion/baseline_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-qk-norm-rope-fusion/baseline_pareto.png \
  --title 'Gemma 4 E2B Baseline Pareto Curve'
```

### 6. Generate The Q/K-Norm + RoPE Pareto Plot

```bash
cd ~/gpu-assignment

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step6-qk-norm-rope-fusion \
  --pattern 'qk_norm_rope_fusion_c*' \
  --output-csv ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_pareto.png \
  --title 'Gemma 4 E2B Q/K RMSNorm + RoPE Fusion Pareto Curve'
```

## Expected Artifacts

The important artifacts for this experiment are:

- `baseline_summary.csv`
- `baseline_pareto.png`
- `qk_norm_rope_fusion_summary.csv`
- `qk_norm_rope_fusion_pareto.png`
- matching `nsys` traces for baseline and qk-norm-rope-fusion

## Success Criteria

The experiment is considered promising only if all of the following hold:

- the server starts cleanly with `--gemma4-kernel-experiment qk-norm-rope-fusion`
- the selected mode still appears in logs
- the model responds coherently
- there is no obvious correctness regression relative to baseline
- AIPerf shows a measurable throughput and/or latency improvement
- CUDA graphs and async scheduling remain enabled unless intentionally disabled for a specific trace
- profiling evidence suggests the fused path is actually active

## Follow-On Candidate

If this experiment does not move the Pareto curve enough, the next candidate remains **PLE branch fusion**.
