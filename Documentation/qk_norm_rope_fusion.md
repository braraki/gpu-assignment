# Gemma 4 Q/K RMSNorm + RoPE Fusion Notes

## Goal

Evaluate the existing `fused_qk_norm_rope` infrastructure on `Gemma 4 E2B` in two separate experiment modes:

- `qk-norm-rope-fusion-lt-512`: control experiment that fuses only supported sub-`512` attention signatures
- `qk-norm-rope-fusion-512`: CUDA-only follow-up that also enables `head_dim=512` fusion

The important framing is:

- `lt-512` remains the lower-risk partial-layer control
- `512` is a separate end-to-end experiment, not a widening of the control mode
- switching kernel revisions or experiment modes requires a fresh torch compile cache

## Why This Experiment Exists

This remains the most direct follow-on to decoder residual fusion because `vLLM` already contains:

- a fused `fused_qk_norm_rope` custom op
- a compile pass that matches unfused Q/K RMSNorm + RoPE
- kernel tests and compile-pass tests for that path

That means the work is mostly:

- extending the CUDA kernel to cover `head_dim=512`
- gating the compile pass correctly per experiment mode
- benchmarking the new end-to-end mode cleanly against baseline and `lt-512`

## Gemma 4 Attention Layout

`Gemma 4 E2B` does not use one uniform attention signature across all decoder layers. In practice the relevant signatures are:

- `(256, 8, 1)`
- `(512, 8, 1)`

That means:

- `qk-norm-rope-fusion-lt-512` should fuse only the `256` family
- `qk-norm-rope-fusion-512` should fuse both the `256` and `512` families on CUDA

The compile pass now registers one pattern per discovered attention signature and guards each rewrite so a `256` registration cannot rewrite a `512` site, and vice versa.

## Public Experiment Modes

The public interface stays behind the same single Gemma 4 flag:

```bash
--gemma4-kernel-experiment baseline
--gemma4-kernel-experiment qk-norm-rope-fusion-lt-512
--gemma4-kernel-experiment qk-norm-rope-fusion-512
```

Mode semantics:

- `baseline`: existing decoder behavior
- `qk-norm-rope-fusion-lt-512`: enables the fusion pass but only registers supported signatures below `512`
- `qk-norm-rope-fusion-512`: CUDA-only mode that enables the same pass machinery and allows `512`-dim fusion as well

Both fusion modes automatically enable:

- `pass_config.enable_qk_norm_rope_fusion = True`
- `+rms_norm`
- `+rotary_embedding`

No extra manual `compilation-config` overrides are needed beyond the standard step-3 launch command.

## Cache Hygiene

`torch.compile` artifacts are keyed separately from the server launch command, so a stale cache can hide a new kernel or pass change. Before switching:

- between `lt-512` and `512`
- after pulling a new `vllm` revision
- after changing kernel heuristics

clear the compile cache:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

If you do not clear it, the server can load an older compiled graph that still contains the previous fused call pattern.

## Commands To Run The Experiments

This section assumes:

- `vllm` and `AIPerf` run on the EC2 instance
- the model is `google/gemma-4-E2B-it`
- benchmark artifacts live under `~/gpu-assignment-results/step6-qk-norm-rope-fusion`
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

### 3. Start The LT-512 Control Server

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
  --gemma4-kernel-experiment qk-norm-rope-fusion-lt-512
```

### 4. Run The LT-512 Control Sweep

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== LT-512 Q/K-norm + RoPE fusion concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_lt_512_c${C}
done
```

### 5. Start The CUDA 512 Fusion Server

Stop the `lt-512` server, then restart:

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
  --gemma4-kernel-experiment qk-norm-rope-fusion-512
```

Expected startup logs should now include both Gemma 4 signatures:

- `(256, 8, 1)`
- `(512, 8, 1)`

### 6. Run The CUDA 512 Sweep

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== CUDA 512 Q/K-norm + RoPE fusion concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_512_c${C}
done
```

### 7. Generate Pareto Plots

Baseline:

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

LT-512 control:

```bash
cd ~/gpu-assignment

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step6-qk-norm-rope-fusion \
  --pattern 'qk_norm_rope_fusion_lt_512_c*' \
  --output-csv ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_lt_512_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_lt_512_pareto.png \
  --title 'Gemma 4 E2B LT-512 Q/K RMSNorm + RoPE Fusion Pareto Curve'
```

CUDA 512 experiment:

```bash
cd ~/gpu-assignment

python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step6-qk-norm-rope-fusion \
  --pattern 'qk_norm_rope_fusion_512_c*' \
  --output-csv ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_512_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-qk-norm-rope-fusion/qk_norm_rope_fusion_512_pareto.png \
  --title 'Gemma 4 E2B CUDA 512 Q/K RMSNorm + RoPE Fusion Pareto Curve'
```

### 8. Generate A Combined Comparison Plot

```bash
cd ~/gpu-assignment

mkdir -p ~/gpu-assignment-results/step6-comparison

python3 plot_aiperf_pareto.py \
  --series 'baseline=~/gpu-assignment-results/step6-qk-norm-rope-fusion::baseline_c*' \
  --series 'qk-norm-rope-fusion-lt-512=~/gpu-assignment-results/step6-qk-norm-rope-fusion::qk_norm_rope_fusion_lt_512_c*' \
  --series 'qk-norm-rope-fusion-512=~/gpu-assignment-results/step6-qk-norm-rope-fusion::qk_norm_rope_fusion_512_c*' \
  --output-csv ~/gpu-assignment-results/step6-comparison/combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/combined_pareto.png \
  --title 'Gemma 4 E2B Q/K RMSNorm + RoPE Experiment Comparison'
```

## Expected Artifacts

The important artifacts are:

- `baseline_summary.csv`
- `baseline_pareto.png`
- `qk_norm_rope_fusion_lt_512_summary.csv`
- `qk_norm_rope_fusion_lt_512_pareto.png`
- `qk_norm_rope_fusion_512_summary.csv`
- `qk_norm_rope_fusion_512_pareto.png`
- `combined_summary.csv`
- `combined_pareto.png`
- matching `nsys` traces for baseline, `lt-512`, and `512`

## Success Criteria

The `512` experiment is considered successful only if all of the following hold:

- the server starts cleanly with `--gemma4-kernel-experiment qk-norm-rope-fusion-512`
- startup logs show both `(256, 8, 1)` and `(512, 8, 1)` are registered
- there is no `Unsupported head dimension for fusedQKNormRope: 512` runtime failure
- direct kernel tests and compile-pass tests still pass
- model responses remain coherent
- there is no obvious correctness regression versus baseline
- AIPerf shows no regression versus `lt-512`, with whole-model improvement as the target
- CUDA graphs and async scheduling remain enabled unless intentionally disabled for tracing

## Follow-On Candidate

If the `512` path is correct but not competitive, the next step is retuning the `SM89` `token_heads_per_warp` thresholds on L4 using forced `1`, `2`, and `4` overrides and then updating the host auto-selection branch with the measured best cutovers.
