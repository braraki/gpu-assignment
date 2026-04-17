# LT-512 Q/K RMSNorm + RoPE Fusion Notes

## Goal

Evaluate a second `Gemma 4 E2B` kernel experiment in `vLLM` using the existing fused Q/K RMSNorm + RoPE infrastructure, but only for Gemma 4 attention layers whose head dimension is below `512`.

This experiment is intended to be:

- lower-risk than a new Gemma-4-only kernel
- compatible with CUDA graphs
- compatible with async scheduling
- easy to benchmark with the existing one-flag `AIPerf` workflow
- explicit about being a **partial-layer** optimization rather than a whole-model one

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

- `qk-norm-rope-fusion-lt-512`

The public interface remains the same single flag used for the other Gemma 4 experiments:

```bash
--gemma4-kernel-experiment qk-norm-rope-fusion-lt-512
```

When this mode is selected, the server should automatically enable the compile-time machinery needed for the existing fusion path:

- `pass_config.enable_qk_norm_rope_fusion = True`
- `+rms_norm`
- `+rotary_embedding`

No manual `compilation-config` changes should be required beyond the existing step-3 launch command.

The important scope limitation is:

- Gemma 4 layers with `head_dim=256` are eligible for this experiment
- Gemma 4 layers with `head_dim=512` are not

So this mode should be understood as **kernel fusion for lt-512 Gemma 4 layers**, not a full-model Q/K fusion mode.

## Gemma 4 Specific Caveat And Fix

The first attempt at this experiment failed on `Gemma 4 E2B` even though the generic fusion pass already existed.

The root cause was that `Gemma 4` does **not** use one uniform attention layout across all decoder layers:

- some layers use the standard `head_dim`
- full-attention layers can use `global_head_dim`
- KV head counts can also differ across layer families

The original `QKNormRoPEFusionPass` incorrectly registered patterns using only the **first** discovered `Attention` layer. That worked for models with one uniform attention signature, but it broke on Gemma 4 when the pass tried to match a pattern traced with one QKV split size against a different layer with a wider QKV projection. The runtime symptom looked like:

```text
Split sizes add up to 2560 but got the tensor's size of 5120
```

The fix was to change the pass so it registers one fusion pattern for **each unique attention signature** discovered in the model:

- `(head_dim, num_heads, num_kv_heads)` per discovered `Attention` layer

That makes the experiment compatible with Gemma 4's mixed attention stack while keeping the implementation generic.

After that, a second issue became clear: the fused CUDA op itself only supports `head_dim` values `64`, `128`, and `256`. Gemma 4 full-attention layers use `head_dim=512`, so the experiment had to be narrowed.

The current behavior is:

- register and run the fused path only for supported lt-512 signatures
- skip unsupported `512`-dim signatures instead of crashing server startup

This means the current experiment measures the effect of partial fusion on the supported Gemma 4 layer family. A follow-up experiment can target `head_dim=512` support separately.

There is now also a regression test covering a mixed-signature attention model so this specific failure mode stays covered.

Before rerunning this experiment on EC2, make sure your `~/vllm` checkout includes that fix. If you ran the experiment before pulling the latest code, update the checkout first and then restart the server.

## Commands To Run The Experiment

This section assumes:

- the `vllm` server and `AIPerf` are both run on the EC2 instance
- the model is `google/gemma-4-E2B-it`
- benchmark artifacts are written under `~/gpu-assignment-results/step6-qk-norm-rope-fusion-lt-512`
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
mkdir -p ~/gpu-assignment-results/step6-qk-norm-rope-fusion-lt-512
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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion-lt-512/baseline_c${C}
done
```

### 3. Start The LT-512 Q/K-Norm + RoPE Fusion Server

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
  --gemma4-kernel-experiment qk-norm-rope-fusion-lt-512
```

### 4. Run The LT-512 Q/K-Norm + RoPE AIPerf Sweep

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
    --artifact-dir ~/gpu-assignment-results/step6-qk-norm-rope-fusion-lt-512/qk_norm_rope_fusion_lt_512_c${C}
done
```

### 5. Generate A Combined Pareto Plot

```bash
cd ~/gpu-assignment
python3 -m pip install -r requirements.txt

python3 plot_aiperf_pareto.py \
  --series 'baseline=~/gpu-assignment-results/step6-decoder-residual-fusion::baseline_c*' \
  --series 'decoder-residual-fusion=~/gpu-assignment-results/step6-decoder-residual-fusion::decoder_residual_fusion_c*' \
  --series 'qk-norm-rope-fusion-lt-512=~/gpu-assignment-results/step6-qk-norm-rope-fusion-lt-512::qk_norm_rope_fusion_lt_512_c*' \
  --output-csv ~/gpu-assignment-results/step6-comparison/combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/combined_pareto.png \
  --title 'Gemma 4 E2B Kernel Experiment Comparison'
```

## Expected Artifacts

The important artifacts for cross-experiment comparison are:

- `combined_summary.csv`
- `combined_pareto.png`
- `combined_pareto_total_tokens.png`
- matching `nsys` traces for baseline and `qk-norm-rope-fusion-lt-512`

## Success Criteria

The experiment is considered promising only if all of the following hold:

- the server starts cleanly with `--gemma4-kernel-experiment qk-norm-rope-fusion-lt-512`
- the selected mode still appears in logs
- the model responds coherently
- there is no obvious correctness regression relative to baseline
- AIPerf shows a measurable throughput and/or latency improvement
- CUDA graphs and async scheduling remain enabled unless intentionally disabled for a specific trace
- profiling evidence suggests the fused path is actually active on the supported lt-512 layer family
- the write-up is explicit that `512`-dim Gemma 4 layers remain unfused in this experiment

## Follow-On Candidate

The natural follow-on to this experiment is a dedicated **`head_dim=512` Q/K RMSNorm + RoPE kernel support** effort. If that is not pursued next, the other major candidate remains **PLE branch fusion**.
