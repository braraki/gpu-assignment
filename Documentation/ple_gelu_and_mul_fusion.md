# PLE GELU x Mul Fusion Notes

## Goal

Evaluate a narrow Step 6 Gemma4 kernel experiment that fuses only the PLE activation path:

1. `gate = per_layer_input_gate(hidden_states)`
2. `gelu(gate, approximate="tanh")`
3. `gated_per_layer = gate * per_layer_input`

The experiment mode is:

```bash
--gemma4-kernel-experiment ple-gelu-and-mul-fusion
```

This mode replaces only the `gelu(gate) * per_layer_input` portion of the PLE branch with a two-input custom op:

```text
ple_gelu_tanh_and_mul(gate, per_layer_input)
```

Out of scope for this experiment:

- fusing `per_layer_input_gate`
- fusing `per_layer_projection`
- fusing `post_per_layer_input_norm`
- fusing the final `hidden_states + per_layer_contribution`

## Why This Is Worth Testing

The baseline `nsys` trace already suggests there is still small unfused elementwise work in the decode hot path that is consistent with the Gemma4 PLE tail.

From [results/decoder_residual_baseline_nsys.txt](/Users/brandonaraki/projects/gpu-assignment/results/decoder_residual_baseline_nsys.txt:63):

- standalone GELU kernel: `19,246` launches and `0.3%` GPU time at [line 77](/Users/brandonaraki/projects/gpu-assignment/results/decoder_residual_baseline_nsys.txt:77)
- standalone BF16 binary elementwise kernel: `18,966` launches and `0.2%` GPU time at [line 78](/Users/brandonaraki/projects/gpu-assignment/results/decoder_residual_baseline_nsys.txt:78)
- small `gemvx` bucket with the same launch count: `18,966` launches and `1.0%` GPU time at [line 67](/Users/brandonaraki/projects/gpu-assignment/results/decoder_residual_baseline_nsys.txt:67)

That evidence is **indirect**. The baseline trace does not expose a dedicated PLE NVTX scope, so those kernels cannot be attributed to PLE with certainty from the existing trace alone.

This experiment fixes that by adding explicit Gemma4 PLE profiling scopes:

- `gemma4.decoder.ple_gelu_and_mul:baseline`
- `gemma4.decoder.ple_gelu_and_mul:ple_gelu_and_mul_fusion`

That makes the follow-up steady-state `nsys` runs decision-useful.

## Why A New Two-Input Kernel Instead Of `cat + GeluAndMul`

The existing `gelu_tanh_and_mul` op in vLLM expects a single packed `[..., 2d]` tensor. Gemma4 PLE has two already-materialized tensors:

- `gate`
- `per_layer_input`

Using the old op would require:

1. `torch.cat([gate, per_layer_input], dim=-1)`
2. calling the existing packed activation kernel

That adds an extra allocation/copy step to save two small elementwise kernels, which weakens the optimization. The new two-input op removes that packing cost and directly computes:

```text
gelu_tanh(gate) * per_layer_input
```

## Modes

- `baseline`
- `ple-gelu-and-mul-fusion`

Only the experiment flag should differ between end-to-end AIPerf runs.

## Scripts

These scripts live under:

```text
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/
```

### 1. Start The Server

Baseline:

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh baseline
```

Fusion:

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/serve_gemma4_experiment.sh ple-gelu-and-mul-fusion
```

### 2. Run The AIPerf Sweep

Baseline:

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh baseline
```

Fusion:

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_aiperf_sweep.sh ple-gelu-and-mul-fusion
```

Artifacts:

- baseline: `~/gpu-assignment-results/step6-baseline/baseline_c*`
- fusion: `~/gpu-assignment-results/step6-ple-gelu-and-mul-fusion/ple_gelu_and_mul_fusion_c*`

### 3. Run The Microbenchmark

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_microbenchmarks.sh
```

This writes:

- `ple_gelu_and_mul_benchmark.csv`
- one PNG per benchmarked `ple_dim`

The microbenchmark compares:

- `native_separate`
- `native_compiled`
- `cat_plus_existing_custom`
- `custom_two_input`

### 4. Run The Steady-State `nsys` Protocol

```bash
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh baseline
gpu-assignment/scripts/step6_ple_gelu_and_mul_fusion/run_nsys_protocol.sh ple-gelu-and-mul-fusion
```

## Plotting

The AIPerf plotting helper now recognizes:

```bash
--experiment ple-gelu-and-mul-fusion
```

Example combined comparison:

```bash
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results \
  --experiment baseline \
  --experiment decoder-residual-fusion \
  --experiment ple-gelu-and-mul-fusion \
  --output-csv ~/gpu-assignment-results/step6-comparison/ple_combined_summary.csv \
  --output-figure ~/gpu-assignment-results/step6-comparison/ple_combined_pareto.png \
  --title 'Gemma 4 E2B Kernel Experiment Comparison'
```

## What To Look For In `nsys`

With `VLLM_NVTX_SCOPES_FOR_PROFILING=1` or `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`, compare:

- `gemma4.decoder.ple_gelu_and_mul:baseline`
- `gemma4.decoder.ple_gelu_and_mul:ple_gelu_and_mul_fusion`

The expected changes are:

- fewer standalone elementwise kernels in the PLE tail
- reduced launch count and bubble time in that scope
- no correctness regression
- no obvious regression in compiled serving throughput
