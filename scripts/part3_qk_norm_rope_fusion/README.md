# Part 3 Q/K Norm + RoPE + VNorm Fusion Scripts

This directory contains the run harness for the Part 3 attention-prep fusion
experiment family.

The `qkv-norm-rope-vnorm-fusion` experiment now uses a Triton-registered
Python custom op, so iterating on the Part 3 kernel does not require rebuilding
the native `_C` extension.

- `serve_qk_norm_rope_fusion_gemma4.sh`: start a Gemma 4 server for a selected Part 3 experiment mode
- `run_aiperf_sweep.sh`: run the standard `AIPerf` concurrency sweep against the current server
- `run_aiperf_c4_load.sh`: run sustained `c=4` load for `nsys` capture
- `run_microbenchmark.sh`: run the operator-only attention-prep microbenchmark
- `dump_microbenchmark_compiled_code.sh`: dump Torch/Inductor-generated code for the isolated compiled providers
- `profile_qkv_norm_rope_vnorm_gemma4_nsys.sh`: profile a selected Part 3 server mode under a rich `nsys` trace
- `process_nsys_report.sh`: export `nsys stats` summaries for a named report
