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
- Framed a GraphNet-style compiler workload path by mapping computation-graph benchmark needs to dense, normalization, and sparse-ish gather/scatter operator primitives on C500.

## Stronger Bullets After Next Phase

Use these only after mini GraphNet/static-inference work is completed:

- Implemented a mini GraphNet-style graph replay harness in Paddle, comparing dynamic graph execution against static/Paddle Inference export on MetaX C500.
- Profiled dense and sparse-ish graph blocks across matmul, layernorm, gather, and scatter, identifying operator-level bottlenecks relevant to tensor compiler benchmarks.
- Evaluated the current C500 Paddle image's suitability for compiler benchmarking, finding that it supports runtime/operator/inference analysis but cannot directly run CINN acceleration because CINN is not compiled in.

## Interview Story

A good 60-second version:

> I wanted to connect my PaddlePaddle internship experience with inference and compiler systems, so I investigated Paddle on a MetaX C500 accelerator. First I built backend probes to see how the vendor image exposes the card to Paddle. It turned out to use Paddle's CUDA-compatible `gpu:0` route, not custom-device dispatch, and CINN was not compiled in. Then I validated Paddle Inference availability and core op coverage, including fp16/fp32 matmul, conv2d, softmax, layernorm, gather, and scatter. The GraphNet angle is that PaddlePaddle/GraphNet is a computation graph dataset for tensor compiler research, so the next step is to replay GraphNet-style graph blocks and compare dynamic vs static/Paddle Inference execution. This makes the project about framework backend behavior and compiler workload readiness, not just running a Paddle demo.

## Evidence Links In This Repo

| Evidence | File |
| --- | --- |
| Paddle backend/device probe script | `scripts/paddle_backend_probe.py` |
| Paddle op probe script | `scripts/paddle_op_probe.py` |
| Paddle backend raw output | `raw/metax-c500-paddle/paddle_backend_probe.json` |
| Paddle op raw output | `raw/metax-c500-paddle/paddle_op_probe.json` |
| Technical note | `notes/metax-c500-paddle-backend-graphnet-probe.md` |

## What Not To Overclaim Yet

Do not claim yet:

- completed full PaddlePaddle/GraphNet benchmark on C500
- measured CINN speedup on C500
- performed kernel-level profiling with vendor profiler
- achieved production inference optimization

Current true claim:

- completed backend capability and op coverage probe
- identified image-level compiler limitation (`CINN=False`)
- established a concrete GraphNet-style inference/compiler workload plan

## Next Resume-Strengthening Work

The highest-return next tasks:

1. Read GraphNet data format and benchmark driver.
2. Run or approximate one GraphNet sample in Paddle.
3. Build mini graph replay blocks: dense, conv, normalization, gather/scatter, mixed.
4. Compare dynamic graph vs static/Paddle Inference export/load.
5. Add repeated-run statistics and `mx-smi` sampling.
