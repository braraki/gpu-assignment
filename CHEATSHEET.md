# EC2 Basics

Replace these placeholders with your real values:

- `<KEY_PATH>`: path to your `.pem` file on your laptop
- `<PUBLIC_DNS_OR_IP>`: the EC2 public DNS name or public IP

Example:

```text
<KEY_PATH>=~/.ssh/gpu-assignment-key.pem
<PUBLIC_DNS_OR_IP>=ec2-12-34-56-78.us-west-2.compute.amazonaws.com
```

## SSH Into The EC2 Instance

Run this once on your laptop if needed:

```bash
chmod 400 <KEY_PATH>
```

Then connect:

```bash
ssh -i <KEY_PATH> ubuntu@<PUBLIC_DNS_OR_IP>
```

## Go To The vLLM Checkout

On the EC2 instance:

```bash
cd ~/vllm
source .venv/bin/activate
```

## Quick Health Check

```bash
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
```

## Start vLLM

Run this on the EC2 instance from `~/vllm` with the virtualenv activated:

```bash
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
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Check That vLLM Is Running

Open a second SSH session and run:

```bash
curl http://localhost:8000/v1/models
```

If the server is healthy, you should get a JSON response listing the model.

## Send A Test Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "In one short paragraph, explain what a GPU does."}
    ],
    "temperature": 0,
    "max_tokens": 120
  }'
```

## Install AIPerf

Run this on the EC2 instance:

```bash
python3 -m venv ~/aiperf-venv
source ~/aiperf-venv/bin/activate
pip install --upgrade pip
pip install aiperf
aiperf --help
```

## Run One AIPerf Benchmark

```bash
source ~/aiperf-venv/bin/activate
mkdir -p ~/gpu-assignment-results/step4-aiperf

aiperf profile \
  --model google/gemma-4-E2B-it \
  --tokenizer google/gemma-4-E2B-it \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint v1/chat/completions \
  --streaming \
  --ui simple \
  --concurrency 1 \
  --request-count 128 \
  --warmup-request-count 8 \
  --synthetic-input-tokens-mean 512 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 0 \
  --artifact-dir ~/gpu-assignment-results/step4-aiperf/baseline_c1
```

## Run The Concurrency Sweep

```bash
source ~/aiperf-venv/bin/activate

for C in 1 2 4 8; do
  echo "=== Running concurrency ${C} ==="
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
    --artifact-dir ~/gpu-assignment-results/step4-aiperf/baseline_c${C}
done
```

## Most Important Result Files

After each run, look in:

```bash
~/gpu-assignment-results/step4-aiperf/baseline_c1
```

The key files are:

- `profile_export_aiperf.json`
- `profile_export_aiperf.csv`

## Plot The Pareto Curve

From this repo on the EC2 instance:

```bash
cd ~/gpu-assignment
python3 plot_aiperf_pareto.py \
  --results-root ~/gpu-assignment-results/step4-aiperf \
  --output-csv ~/gpu-assignment-results/step4-aiperf/baseline_summary.csv \
  --output-figure ~/gpu-assignment-results/step4-aiperf/baseline_pareto.png
```

## If The Public IP Changes

After an EC2 stop/start cycle, the public IP or DNS may change. Get the new value from the AWS console and use it in the SSH command again.
