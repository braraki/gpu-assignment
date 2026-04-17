# Plotting AIPerf Results

This file explains how to run the plotting helper after you have already produced the benchmark runs:

- `baseline_c1`
- `baseline_c2`
- `baseline_c4`
- `baseline_c8`

## 1. Clone This Repo On The EC2 Instance

Use your fork or remote URL:

```bash
git clone <your-gpu-assignment-repo-url>
cd gpu-assignment
```

## 2. Install The Plotting Dependency

The plotting script only needs `matplotlib` beyond the Python standard library.

Install it with:

```bash
python3 -m pip install -r requirements.txt
```

## 3. Confirm Your AIPerf Results Exist

The plotting script expects AIPerf artifact directories like:

```text
~/gpu-assignment-results/step4-aiperf/baseline_c1/profile_export_aiperf.json
~/gpu-assignment-results/step4-aiperf/baseline_c2/profile_export_aiperf.json
~/gpu-assignment-results/step4-aiperf/baseline_c4/profile_export_aiperf.json
~/gpu-assignment-results/step4-aiperf/baseline_c8/profile_export_aiperf.json
```

The script scans all directories matching `baseline_c*` under the results root.

## 4. Run The Plotting Script

From the cloned repo root:

```bash
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step4-aiperf \
  --output-csv ~/gpu-assignment-results/step4-aiperf/baseline_summary.csv \
  --output-figure ~/gpu-assignment-results/step4-aiperf/baseline_pareto.png
```

## 5. What The Script Produces

It writes:

- `baseline_summary.csv`
- `baseline_pareto.png`
- `baseline_pareto_total_tokens.png`

The CSV has one row per run and includes:

- concurrency
- output tokens/s/user
- output tokens/s/gpu
- output token throughput
- total tokens/s/user
- total tokens/s/gpu
- total token throughput
- TTFT
- ITL
- request latency

The first PNG is the output-token Pareto curve:

- x-axis: `output tokens/s/user`
- y-axis: `output tokens/s/gpu`

The second PNG is the total-token Pareto curve:

- x-axis: `total tokens/s/user`
- y-axis: `total tokens/s/gpu`

## 6. Optional Arguments

If you need them:

```bash
python3 plot_aiperf_pareto.py --help
```

Useful options:

- `--pattern` to change which run directories are scanned
- `--gpu-count` if you later benchmark with more than one GPU
- `--title` to override the plot title

## 7. Typical Workflow

1. Start `vllm`.
2. Run the AIPerf concurrency sweep.
3. Confirm each `baseline_c*` directory contains `profile_export_aiperf.json`.
4. Run the plotting script.
5. Inspect `baseline_summary.csv`, `baseline_pareto.png`, and `baseline_pareto_total_tokens.png`.
