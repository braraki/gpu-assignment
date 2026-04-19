# GPU Assignment Notes

This repo now has separate step documents:

- `Part 1 Benchmarking.md`: setup and results template for the first vanilla `vllm` + `nsys` benchmark run
- `Part-2-decoder-fusion.md`: decoder residual fusion report, source mapping, Part 1 baseline references, and expected changes
- `Part-3-qk-norm-rope-fusion.md`: attention-prep fusion report, source mapping, Part 1 baseline references, and the new QKV fusion plan
- `Part-4-qkv-norm-rope-vnorm-kvcache-fusion.md`: full post-GEMM attention-prep-and-cache fusion report, source mapping, and Part 4 run instructions
- `scripts/part2_decoder_fusion/`: Part 2 run scripts for the decoder-residual-fusion benchmark and `nsys`
- `scripts/part3_qk_norm_rope_fusion/`: Part 3 run scripts for the QKV attention-prep fusion benchmark family
- `scripts/part4_qkv_norm_rope_vnorm_kvcache_fusion/`: Part 4 run scripts for the full post-GEMM QKV + KV-cache fusion benchmark family
- `Step3.md`: getting `Gemma 4 E2B` running on the EC2 instance
- `Step4.md`: benchmarking the running server and building the step-4 Pareto curve
- `Step5.md`: collecting and comparing Nsight Systems GPU traces
- `Documentation/decoder_residual_fusion.md`: step-6 notes, experiment design, and decoder residual fusion targets
- `Documentation/async_output_sync_reduction.md`: step-6 notes for the async scheduling output handoff experiment that targets `cudaEventSynchronize`
- `Documentation/qk_norm_rope_fusion.md`: step-6 follow-on notes for the QKV attention-prep fusion direction
- `CHEATSHEET.md`: quick EC2 commands for SSH, `vllm`, requests, and `AIPerf`
- `PLOTTING.md`: how to install the plotting dependency and generate the Pareto figure
- `requirements.txt`: minimal dependency file for the plotting helper

Start with `Step3.md` if you need to re-create the server setup.

Use `Step4.md` now that the model is up and responding.
