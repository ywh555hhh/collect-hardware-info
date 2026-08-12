# Resume Draft: Paddle / C500 / GraphNet Direction

Date: 2026-08-12

This file turns the current Paddle-on-C500 evidence into resume-ready language. It should be updated as the project deepens.

## Recommended Project Title

**PaddlePaddle Backend and GraphNet-Style Inference Probe on MetaX C500**

Alternative shorter title:

**Paddle GraphNet Workload Probe on C500**

## One-Line Positioning

Investigated PaddlePaddle runtime and operator behavior on a non-NVIDIA MetaX C500 accelerator, connecting backend dispatch, Paddle Inference, and GraphNet-style computation graph workloads for AI compiler / inference-engine evaluation.

## Resume Bullets: Current Evidence Level

These are safe to use now, based on current repository evidence:

- Built reproducible PaddlePaddle probes on MetaX C500 to inspect framework backend routing, device discovery, Paddle Inference availability, and core operator coverage.
- Identified that the C500 Paddle image exposes the accelerator through Paddle's CUDA-compatible `gpu:0` path (`paddlepaddle-gpu 2.6.0+maca3.0.0.5`) rather than a Paddle custom-device backend; verified `is_compiled_with_cuda=True`, `is_compiled_with_cinn=False`, and no custom device types.
- Validated GPU execution for fp32/fp16 matmul, conv2d, softmax, layernorm, gather, and scatter on `Place(gpu:0)`, with `paddle.utils.run_check()` passing and Paddle Inference APIs available.
- Implemented a mini GraphNet-style graph replay harness in Paddle, comparing dynamic graph execution against `paddle.jit.save/load` static graph execution on MetaX C500.
- Profiled dense and sparse-ish graph blocks across matmul, gelu, layernorm, gather, scatter, projection, and reduction; observed dynamic avg latency of 0.333 ms for dense block, 0.878 ms for gather/scatter block, and 1.268 ms for mixed graph block on `Place(gpu:0)`.
- Verified JIT export/load for dense and mixed graph blocks, with dense block improving from 0.333 ms dynamic to 0.255 ms JIT path, and mixed block from 1.268 ms dynamic to 1.122 ms JIT path.
- Exported fixed-shape dense and mixed graph blocks to Paddle Inference and validated `PaddleInferPredictor` execution on C500, observing 1.515 ms and 2.670 ms average end-to-end predictor latency for the two blocks.
- Brought up an official PaddlePaddle/GraphNet `ernie-3.0-nano-zh` sample on C500: direct dygraph forward succeeded on `Place(gpu:0)` with output shape `[1, 312]`; official benchmark static paths still fail in `paddle.jit.to_static`, isolating the next compatibility gap.
- Identified official GraphNet bring-up compatibility gaps on C500: Paddle accepts `gpu` / `gpu:0` but rejects GraphNet's expected `cuda` / `cuda:0` device strings, and the upstream benchmark has a backend import path mismatch requiring a thin wrapper or patch.
- Evaluated the current C500 Paddle image's suitability for compiler benchmarking, finding that it supports runtime/operator/static-graph analysis but cannot directly run CINN acceleration because CINN is not compiled in.

## Stronger Bullets After Next Phase

Use these only after real GraphNet benchmark or vendor-profiler work is completed:

- Ran a real PaddlePaddle/GraphNet sample on C500 and mapped its graph operators to backend kernels and runtime paths.
- Used vendor profiling tools to identify kernel-level bottlenecks in sparse indexing, memory movement, and static graph replay.
- Compared Paddle dynamic graph, static graph, and Paddle Inference predictor execution on the same exported workload.

## Interview Story

A good 60-second version:

> I wanted to connect my PaddlePaddle internship experience with inference and compiler systems, so I investigated Paddle on a MetaX C500 accelerator. First I built backend probes to see how the vendor image exposes the card to Paddle. It turned out to use Paddle's CUDA-compatible `gpu:0` route, not custom-device dispatch, and CINN was not compiled in. Then I validated Paddle Inference availability and core op coverage, including fp16/fp32 matmul, conv2d, softmax, layernorm, gather, and scatter. The GraphNet angle is that PaddlePaddle/GraphNet is a computation graph dataset for tensor compiler research, so I built a mini GraphNet-style Paddle workload with dense, sparse gather/scatter, mixed graph, and readout blocks, then compared dynamic graph execution with `paddle.jit.save/load` static graph execution. This makes the project about framework backend behavior and compiler workload readiness, not just running a Paddle demo.

## Evidence Links In This Repo

| Evidence | File |
| --- | --- |
| Paddle backend/device probe script | `scripts/paddle_backend_probe.py` |
| Paddle op probe script | `scripts/paddle_op_probe.py` |
| Paddle backend raw output | `raw/metax-c500-paddle/paddle_backend_probe.json` |
| Paddle op raw output | `raw/metax-c500-paddle/paddle_op_probe.json` |
| Mini GraphNet-style workload script | `scripts/paddle_mini_graphnet_probe.py` |
| Mini GraphNet-style raw output | `raw/metax-c500-paddle/mini_graphnet/mini_graphnet_probe.json` |
| Paddle Inference predictor script | `scripts/paddle_inference_graphnet_probe.py` |
| Paddle Inference predictor raw output | `raw/metax-c500-paddle/paddle_inference_graphnet/paddle_inference_graphnet_probe.json` |
| Paddle device alias probe script | `scripts/paddle_device_alias_probe.py` |
| Paddle device alias raw output | `raw/metax-c500-paddle/device_alias/paddle_device_alias_probe.json` |
| Official GraphNet bring-up script | `scripts/official_graphnet_c500_bringup_probe.py` |
| Official GraphNet raw output | `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json` |
| Paddle `_C_ops` signature script | `scripts/paddle_cops_signature_probe.py` |
| Paddle `_C_ops` signature raw output | `raw/metax-c500-paddle/cops_signature/paddle_cops_signature_probe.json` |
| Official GraphNet alignment note | `notes/paddle-graphnet-official-alignment.md` |
| Technical note | `notes/metax-c500-paddle-backend-graphnet-probe.md` |

## What Not To Overclaim Yet

Do not claim yet:

- completed full PaddlePaddle/GraphNet benchmark on C500
- measured CINN speedup on C500
- performed kernel-level profiling with vendor profiler
- achieved production inference optimization

Current true claim:

- completed backend capability and op coverage probe
- completed mini GraphNet-style dynamic/static graph workload probe
- completed Paddle Inference predictor probe for dense/mixed graph blocks
- completed device-string compatibility probe for official GraphNet bring-up
- completed official GraphNet Paddle sample direct-dygraph bring-up on C500
- identified image-level compiler limitation (`CINN=False`)
- established a concrete path toward real GraphNet and vendor-profiler follow-up

## Next Resume-Strengthening Work

The highest-return next tasks:

1. Read GraphNet data format and benchmark driver.
2. Patch or wrap official GraphNet Paddle benchmark so `cuda` maps to Paddle `gpu:0` on C500.
3. Fix or bypass the official static benchmark `to_static` compatibility issue for generated `_C_ops.full` calls.
4. Add vendor-profiler or `mx-smi` time-series sampling around each graph block.
5. Compare predictor latency under larger graph sizes and batched/repeated request patterns.


## Official Static Patch Follow-Up

A temporary generated-code patch experiment is recorded in `raw/metax-c500-paddle/official_graphnet_static_patch/official_graphnet_static_patch_probe.json` and implemented by `scripts/official_graphnet_static_patch_probe.py`. Rewriting all 159 generated `_C_ops` calls in the official `ernie-3.0-nano-zh` sample to higher-level Paddle APIs allowed the official `compiler=nope` benchmark to complete with `eager:success compiled:success`. The measured e2e median was 4.415 ms for eager and 4.409 ms for compiled; GPU event timing reported 0.0 ms, so GPU-only timing is not reliable in this Paddle/MACA image.
