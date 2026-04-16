# GPU Assignment

This repo documents the assignment setup and the immediate execution plan for steps 2 and 3:

1. Pick a model.
2. Get it running with `vllm serve`.
3. Confirm the server responds correctly.

The later steps, including benchmarking, `nsys` tracing, optimization, and a PR against the `vllm` fork, are intentionally deferred.

## Assignment Overview

The full assignment is:

1. Fork `vllm` or `sglang`.
2. Pick a model that is not too large and works on a single GPU.
3. Start the server and verify it responds correctly.
4. Benchmark it and produce a latency-throughput Pareto curve.
5. Collect GPU traces with `nsys`.
6. Identify and implement a performance optimization and open a PR.
7. Re-benchmark and verify output quality did not regress.

This document only covers the setup and bring-up work needed for steps 2 and 3.

## Current Scope

In scope now:

- AWS EC2 launch and connection
- EC2 machine setup
- `vllm` source checkout and editable install
- `Gemma 4 E2B` best-effort text-only launch on a single `g6.xlarge`
- Basic API validation

Out of scope for now:

- concurrency benchmarking
- `aiperf`
- `nsys`
- code optimization
- pull request preparation
- multimodal validation

## Model Decision

The target model is:

- `google/gemma-4-E2B-it`

Why this model:

- It matches the requested direction to target `Gemma 4`.
- It is the smallest `Gemma 4` instruction-tuned checkpoint discussed in the `vLLM` Gemma 4 recipe.
- It is the most realistic `Gemma 4` candidate to try on a single `g6.xlarge`.

Important constraint:

- This bring-up is intentionally **text-only first**.
- We will launch with `--language-model-only` so the first goal is "server starts and answers text prompts."
- Image and audio support are deferred until the basic text path is confirmed.

## Hardware Risk Note

This plan accepts an explicit memory-fit risk.

- The official `vLLM` Gemma 4 recipe lists `Gemma 4 E2B IT` as needing `1x (24 GB+)` NVIDIA GPU memory in BF16.
- AWS documents `g6.xlarge` as `1 x NVIDIA L4 GPU 22 GiB`.
- That means this configuration is outside the documented BF16 recommendation.

Why proceed anyway:

- `Gemma 4 E2B` is small enough that a reduced-context, reduced-concurrency, text-only launch may still fit.
- We are intentionally lowering memory pressure with:
  - `--language-model-only`
  - smaller `--max-model-len`
  - smaller `--max-num-seqs`
  - smaller `--max-num-batched-tokens`
  - conservative `--gpu-memory-utilization`

Success criterion for this phase:

- The server starts.
- The model loads.
- `/v1/models` responds.
- One text request returns a coherent answer.

Failure criterion for this phase:

- The model still does not fit after the documented fallback ladder below.

## AWS Step-By-Step Setup

This section assumes you are new to AWS and want the exact EC2 console workflow.

### 1. Sign In And Open EC2

1. Sign in to the AWS Console in your browser.
2. In the search bar at the top, type `EC2`.
3. Click `EC2` from the results.
4. Look at the top-right corner and note the currently selected AWS region.

Use a region that offers `g6.xlarge`. If the region you are in does not offer it, switch regions before creating the instance. Try another region first rather than changing instance family.

### 2. Start Launching The Instance

1. In the EC2 console, click `Instances` in the left sidebar.
2. Click `Launch instances`.
3. In `Name and tags`, set the instance name to:

```text
gpu-assignment-g6
```

### 3. Choose The Machine Image

1. In `Application and OS Images (Amazon Machine Image)`, search for an NVIDIA Deep Learning AMI.
2. Prefer an Ubuntu-based NVIDIA Deep Learning AMI if one is available.
3. Choose a recent image rather than an old one.

Reason for this choice:

- It reduces the amount of CUDA and driver setup you need to do by hand.

### 4. Choose The Instance Type

1. In `Instance type`, click the selector.
2. Search for:

```text
g6.xlarge
```

3. Select `g6.xlarge`.

If `g6.xlarge` is unavailable in the selected region, stop here and change regions before continuing.

### 5. Create Or Select A Key Pair

This is what you will use to SSH into the instance from your laptop.

1. In `Key pair (login)`, click `Create new key pair` if you do not already have one you want to use.
2. Use a name like:

```text
gpu-assignment-key
```

3. Choose:
   - `RSA`
   - `.pem`
4. Download the file when prompted.
5. Save it somewhere you will remember.

Important:

- AWS only gives you the private key file once.
- If you lose it, you do not simply re-download it later.
- Treat it like a password.

### 6. Configure Networking

1. In `Network settings`, keep the default VPC unless you already know you need a different one.
2. Make sure the instance gets a public IP address.
3. Under security group settings, allow:
   - `SSH`
   - Source: `My IP`

Do **not** use `Anywhere` / `0.0.0.0/0` for SSH unless you have a specific reason.

### 7. Configure Storage

The model cache, Python environment, and logs will take space.

1. In `Configure storage`, set the root volume to:

```text
150 GiB gp3
```

This is a practical default for:

- the AMI
- a local `vllm` checkout
- Hugging Face cache
- logs and temporary files

### 8. Launch The Instance

1. Review the settings.
2. Click `Launch instance`.
3. Wait for the success page.
4. Click through to the new instance.

### 9. Wait For It To Become Healthy

On the instance details page:

1. Wait until `Instance state` says `Running`.
2. Wait until both status checks pass.
3. Copy either:
   - the `Public IPv4 address`, or
   - the `Public IPv4 DNS`

You will use that value for SSH.

## SSH From Your Laptop

These instructions assume macOS or Linux on your laptop.

### 1. Move The Key File Somewhere Stable

Example:

```bash
mv ~/Downloads/gpu-assignment-key.pem ~/.ssh/gpu-assignment-key.pem
```

### 2. Fix The File Permissions

SSH will reject the key if it is too open.

```bash
chmod 400 ~/.ssh/gpu-assignment-key.pem
```

### 3. Connect To The Instance

Use the public IP or public DNS you copied from the EC2 console:

```bash
ssh -i ~/.ssh/gpu-assignment-key.pem ubuntu@<public-ip-or-dns>
```

Example:

```bash
ssh -i ~/.ssh/gpu-assignment-key.pem ubuntu@ec2-12-34-56-78.us-west-2.compute.amazonaws.com
```

Notes:

- For Ubuntu-based AMIs, `ubuntu` is usually the correct username.
- If `ubuntu` fails, check the AMI documentation in the AWS console and try the documented default username for that image.

## First Checks After Login

Once you are connected over SSH, run:

```bash
nvidia-smi
df -h
pwd
```

What to look for:

- `nvidia-smi` should show one NVIDIA L4 GPU.
- `df -h` should show your root volume and confirm you have enough free disk space.

## EC2 Software Setup

This section assumes you want to work from a source checkout of your forked `vllm`.

### 1. Install Or Confirm Basic Tools

The NVIDIA Deep Learning AMI may already include some tools, but do not assume everything is present.

Check for:

```bash
python3 --version
git --version
curl --version
```

If `uv` is not installed, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc 2>/dev/null || true
source ~/.zshrc 2>/dev/null || true
```

If your shell still does not find `uv`, open a new SSH session and try:

```bash
uv --version
```

### 2. Clone Your Fork Of vLLM

Use your actual fork URL in place of the placeholder below:

```bash
git clone <your-vllm-fork-url>
cd vllm
```

If you want to work from a specific branch, switch to it after cloning.

### 3. Create And Activate A Python 3.12 Environment

If the AMI already has Python 3.12 available:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

If `python3.12` is not available, install it first using the package manager that matches the AMI, then repeat the `uv venv` command.

### 4. Install vLLM In Editable Mode

Use the precompiled path first because it is faster and is enough for bring-up:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Then verify the import:

```bash
python -c "import vllm; print(vllm.__version__)"
```

### 5. Configure Hugging Face Access

You need Hugging Face access to pull `google/gemma-4-E2B-it`.

Install the CLI if needed:

```bash
uv pip install "huggingface_hub[cli]"
```

Then log in:

```bash
huggingface-cli login
```

Paste your Hugging Face token when prompted.

Before continuing, confirm that:

- your token is valid
- your account has access to `google/gemma-4-E2B-it`

## vLLM Launch Plan

This is the first launch command to try.

### 1. Primary Launch Command

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

Why these flags:

- `--language-model-only` avoids multimodal encoder memory use.
- `--max-model-len 4096` reduces KV cache pressure versus the model maximum.
- `--max-num-seqs 8` and `--max-num-batched-tokens 1024` keep concurrency and scheduler memory lower.
- `--gpu-memory-utilization 0.80` leaves some safety margin.
- `--enable-chunked-prefill`, `--async-scheduling`, and `FULL_AND_PIECEWISE` keep the serving path aligned with the later performance work.

### 2. Fallback Ladder If The Model Does Not Fit

If the server fails to start because of memory pressure, retry in this order.

#### Fallback 1

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve google/gemma-4-E2B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --async-scheduling \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

#### Fallback 2

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve google/gemma-4-E2B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --async-scheduling \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

#### Fallback 3

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve google/gemma-4-E2B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.80 \
  --enable-chunked-prefill \
  --async-scheduling \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

If the model still does not fit after fallback 3, stop and record that `g6.xlarge` is not viable for this assignment phase with the current model choice.

## Validation Steps

Once the server is running, open a second SSH session and validate it.

### 1. Confirm The Model Is Registered

```bash
curl http://localhost:8000/v1/models
```

### 2. Send One Deterministic Text Request

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

Success means:

- the request returns HTTP 200
- the output is coherent and legible

### 3. Check The Logs

What you want:

- the model loaded successfully
- the server is serving requests normally
- there is no forced eager-only fallback caused by `--enforce-eager`

What you should not worry about yet:

- exact tokens/sec
- benchmark results
- perfect memory efficiency

## Test Plan For This Phase

AWS checks:

- instance launches successfully
- SSH login works from your laptop
- `nvidia-smi` shows one L4 GPU

Environment checks:

- Hugging Face auth works for `google/gemma-4-E2B-it`
- `python -c "import vllm; print(vllm.__version__)"`

Server checks:

- `/v1/models` responds
- one deterministic text request returns a coherent answer
- the server runs without forcing eager mode

## Safe Stop And Terminate Behavior

When you are done for the day:

- Use `Stop instance` in the EC2 console to avoid paying for compute time while preserving the root disk.

When you are completely done with the machine:

- Use `Terminate instance`.

Be careful:

- `Terminate` is meant to end the instance permanently.
- After a stop/start cycle, the public IP or public DNS can change unless you set up a stable IP separately.

## Monitoring Basics

Useful commands on the instance:

```bash
nvidia-smi
df -h
free -h
```

Useful console actions:

- EC2 -> Instances -> select the instance -> `Instance state`
- choose `Stop instance` when you are not using it
- choose `Terminate instance` only when you are sure you are finished

## Deferred Later Steps

The following are intentionally not part of this document's active execution scope yet:

- benchmarking with different concurrencies
- Pareto curve generation
- `aiperf`
- `nsys`
- sync injection experiments
- performance optimization inside `vllm`
- pull request creation
- output regression checks beyond basic human judgement

## References

- `vLLM` Gemma 4 recipe: <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>
- AWS accelerated instance docs: <https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html>
- Gemma 4 E2B model card: <https://huggingface.co/google/gemma-4-E2B-it>
