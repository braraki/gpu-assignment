# Part 2 Decoder Fusion Scripts

This directory contains the run harness for the Part 2 decoder residual fusion
experiment. Baseline benchmarking and baseline `nsys` capture live in Part 1.

- `serve_decoder_residual_fusion_gemma4.sh`: start the decoder-residual-fusion server
- `run_aiperf_sweep.sh`: run the standard `AIPerf` concurrency sweep against the current server
- `run_aiperf_c4_load.sh`: run sustained `c=4` load for `nsys` capture
- `run_microbenchmark.sh`: run the operator-only decoder residual fusion microbenchmark
- `profile_decoder_residual_fusion_gemma4_nsys.sh`: profile the fusion server under a rich `nsys` trace
- `process_nsys_report.sh`: export `nsys stats` summaries for a named report
