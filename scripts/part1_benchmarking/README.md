# Part 1 Benchmarking Scripts

This directory contains the minimal harness for the first vanilla `vllm` benchmark and `nsys` profiling run.

- `serve_vanilla_gemma4.sh`: start the server without profiling
- `profile_vanilla_gemma4_nsys.sh`: start the server under `nsys`
- `run_aiperf_c4_load.sh`: run an `AIPerf` load that matches the normal concurrency benchmark shape at `c=4`
