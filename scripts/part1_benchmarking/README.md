# Part 1 Benchmarking Scripts

This directory contains the minimal harness for the first vanilla `vllm` benchmark and `nsys` profiling run.

- `serve_vanilla_gemma4.sh`: start the server without profiling
- `profile_vanilla_gemma4_nsys.sh`: start the server under `nsys`
- `run_single_round_load.sh`: send one round of `4` concurrent requests
- `single_round_chat_load.py`: Python client used by the load wrapper
