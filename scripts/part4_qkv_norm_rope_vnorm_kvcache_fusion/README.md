# Part 4 Full Post-GEMM Fusion Scripts

This directory contains the run harness for the Part 4 full post-GEMM fusion
experiment family.

The `qkv-norm-rope-vnorm-kvcache-fusion` experiment uses a Triton-registered
Python custom op, so iterating on the Part 4 kernel does not require rebuilding
the native `_C` extension.

- `serve_qkv_norm_rope_vnorm_kvcache_fusion_gemma4.sh`: start a Gemma 4 server for a selected Part 4 experiment mode
- `run_aiperf_sweep.sh`: run the standard `AIPerf` concurrency sweep against the current server
- `run_aiperf_c4_load.sh`: run sustained `c=4` load for `nsys` capture
- `run_microbenchmark.sh`: run the full post-GEMM operator microbenchmark
- `dump_microbenchmark_compiled_code.sh`: dump Torch/Inductor-generated code for the isolated compiled providers
- `profile_qkv_norm_rope_vnorm_kvcache_fusion_gemma4_nsys.sh`: profile a selected Part 4 server mode under a rich `nsys` trace
- `process_nsys_report.sh`: export `nsys stats` summaries for a named report

The microbenchmark provider names mean:

- `baseline_compiled`: the same unfused post-GEMM block under `torch.compile`
- `post_gemm_kvcache_custom_op`: new Part 4 fused post-GEMM custom op that also writes K/V into KV cache

The generated benchmark plots are grouped bar charts by token count, with one
figure per head dimension.
