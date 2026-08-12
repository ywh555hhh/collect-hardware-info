# MetaX C500 Paddle backend and GraphNet-oriented probe

Date: 2026-08-12

This note starts the PaddlePaddle-side continuation of the C500 investigation. The goal is not to run a generic Paddle demo. The goal is to build a resume-grade story around:

- PaddlePaddle framework/backend knowledge
- C500 runtime behavior under Paddle
- operator/kernel coverage
- Paddle Inference availability
- GraphNet-style computation graph / tensor compiler workloads

## Why GraphNet Matters Here

PaddlePaddle/GraphNet is not a conventional GNN library. It is a large-scale computation graph database and benchmark suite for tensor compiler research. The official repository describes GraphNet as a dataset of 2.7K+ real-world deep learning computation graphs, with benchmark paths for compiler evaluation including PaddlePaddle/CINN and PyTorch/TorchInductor.

That makes it a better fit for the target resume keywords than PaddleCFD or PaddleMaterials demos:

- AI compiler
- graph-level optimization
- kernel/backend coverage
- inference/static graph execution
- framework runtime behavior

## C500 Paddle Image Summary

Raw data:

- `raw/metax-c500-paddle/paddle_backend_probe.json`
- `raw/metax-c500-paddle/paddle_op_probe.json`
- `raw/metax-c500-paddle/device_alias/paddle_device_alias_probe.json`
- `raw/metax-c500-paddle/mini_graphnet/mini_graphnet_probe.json`
- `raw/metax-c500-paddle/paddle_inference_graphnet/paddle_inference_graphnet_probe.json`
- `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json`
- `raw/metax-c500-paddle/cops_signature/paddle_cops_signature_probe.json`

Scripts:

- `scripts/paddle_backend_probe.py`
- `scripts/paddle_device_alias_probe.py`
- `scripts/paddle_op_probe.py`
- `scripts/paddle_mini_graphnet_probe.py`
- `scripts/paddle_inference_graphnet_probe.py`
- `scripts/official_graphnet_c500_bringup_probe.py`
- `scripts/paddle_cops_signature_probe.py`

Environment observed:

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.3 |
| Python | 3.10.10 |
| Paddle package | `paddlepaddle-gpu 2.6.0+maca3.0.0.5` |
| MACA | 3.0.0.8 |
| `mx-smi` | 2.2.6 |
| KMD | 3.8.30 |
| GPU | MetaX C500 sGPU slice |
| sGPU quota | 25% compute / 16GB visible quota |

The important finding: this Paddle image uses Paddle's `gpu:0` route, not a Paddle custom-device route.

| Probe | Result |
| --- | --- |
| `paddle.device.get_device()` | `gpu:0` |
| `paddle.is_compiled_with_cuda()` | `True` |
| `paddle.is_compiled_with_xpu()` | `False` |
| `paddle.is_compiled_with_cinn()` | `False` |
| custom device types | `[]` |
| `maca:0` device string | unsupported |
| `cuda` / `cuda:0` device string | unsupported by Paddle Python API in this image |
| `gpu` / `gpu:0` device string | supported |
| Paddle Inference API | available |
| `paddle.utils.run_check()` | pass |

Interpretation:

- C500 is exposed to Paddle as a CUDA/GPU-compatible backend.
- Official GraphNet Paddle scripts use `--device cuda`, but this image requires Paddle device strings `gpu` or `gpu:0`; official bring-up likely needs a small wrapper or patch.
- The image is not suitable for directly testing Paddle CINN acceleration because CINN is not compiled in.
- The image is suitable for backend/operator coverage, Paddle Inference path checks, dynamic/static graph smoke, and GraphNet-style op mix probing.

## Operator Smoke Results

All of these ran on `Place(gpu:0)`:

| Op | Shape | dtype | avg ms |
| --- | --- | --- | ---: |
| matmul | 1024 x 1024 | fp32 | 0.103 |
| matmul | 1024 x 1024 | fp16 | 0.030 |
| softmax | 4096 x 1024 | fp32 | 0.054 |
| layernorm | 4096 x 1024 | fp32 | 0.048 |
| conv2d | 16 x 64 x 56 x 56 -> 128 channels | fp32 | 0.268 |
| gather | 65536 x 128 from 200000 x 128 | fp32 | 0.119 |
| scatter | 65536 x 128 into 200000 x 128 | fp32 | 0.393 |

The gather/scatter pair is the most relevant for GraphNet-style work because it probes sparse-ish indexing and memory movement. Scatter is noticeably slower than gather, which is consistent with irregular writes and potential write conflicts being harder than reads.

## Mini GraphNet-Style Workload

This is not the official PaddlePaddle/GraphNet benchmark. It is a controlled mini workload that mirrors GraphNet-relevant operator families:

- dense graph block: `matmul -> gelu -> matmul -> layernorm residual`
- sparse-ish block: `gather -> elementwise scale -> scatter`
- mixed block: dense block + sparse block + projection
- graph readout: mixed block + reduction

Parameters:

| Parameter | Value |
| --- | --- |
| nodes | 8192 |
| edges | 65536 |
| hidden | 256 |
| warmup | 10 |
| iterations | 50 |
| device | `gpu:0` |

Dynamic graph results:

| Workload | avg ms | p50 ms | p90 ms | Place |
| --- | ---: | ---: | ---: | --- |
| dense block | 0.333 | 0.332 | 0.336 | `Place(gpu:0)` |
| sparse gather/scatter block | 0.878 | 0.878 | 0.884 | `Place(gpu:0)` |
| mixed graph block | 1.268 | 1.270 | 1.277 | `Place(gpu:0)` |
| graph readout reduce | 1.341 | 1.285 | 1.435 | `Place(gpu:0)` |

`paddle.jit.save/load` results:

| Workload | save/load | avg ms | Place | Observation |
| --- | --- | ---: | --- | --- |
| dense block | pass | 0.255 | `Place(gpu:0)` | faster than dynamic path in this smoke run |
| mixed graph block | pass | 1.122 | `Place(gpu:0)` | faster than dynamic path in this smoke run |

Interpretation:

- The image supports Paddle dynamic graph execution on C500 for dense, sparse-ish, mixed, and reduction graph blocks.
- The image can export and reload these Paddle layers through `paddle.jit.save/load`, so it is useful for static graph/runtime readiness exploration.
- The sparse-ish gather/scatter block is slower than the dense block despite less arithmetic density, which makes it a good follow-up target for memory traffic and irregular-write analysis.
- This still does not prove full official GraphNet benchmark support; it is a controlled approximation that de-risks the backend/operator/static graph path first.

## Paddle Inference Predictor Path

The next probe exports fixed-shape dense and mixed graph blocks to `.pdmodel/.pdiparams`, then loads them through `paddle.inference.Config` and `PaddleInferPredictor` with GPU enabled.

Raw data:

- `raw/metax-c500-paddle/paddle_inference_graphnet/paddle_inference_graphnet_probe.json`
- `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json`
- `raw/metax-c500-paddle/cops_signature/paddle_cops_signature_probe.json`

Parameters match the mini GraphNet-style workload: 8192 nodes, 65536 edges, hidden size 256, 10 warmup iterations, and 50 measured iterations.

| Workload | Predictor path | avg ms | p50 ms | p90 ms | Inputs | Output |
| --- | --- | ---: | ---: | ---: | --- | --- |
| dense block | GPU predictor | 1.515 | 1.502 | 1.555 | `x` | `[8192, 256]` |
| mixed graph block | GPU predictor | 2.670 | 2.664 | 2.691 | `x`, `src`, `dst` | `[8192, 256]` |

Important interpretation:

- Predictor timing includes input handle reshape, CPU-to-predictor feed, `predictor.run()`, and output fetch, so it is an end-to-end deployment-path timing rather than a pure GPU-kernel timing.
- Paddle Inference logs show the analysis predictor running IR passes including matmul mapping/fusion passes, `fc_fuse_pass`, `fc_elementwise_layernorm_fuse_pass`, `auto_mixed_precision_pass`, `inplace_op_var_pass`, `memory_optimize_pass`, and GPU parameter sync.
- The dense predictor graph reports roughly 0.5MB persistable params, while the mixed graph reports roughly 8.8MB persistable params and larger temporary buffers for gather/scatter intermediates.
- This makes the image useful for studying Paddle Inference graph optimization and deployment-path overhead on a non-NVIDIA backend, even without CINN.

## Official GraphNet Sample Bring-Up

A real official PaddlePaddle/GraphNet sample was tested on C500:

- sample: `paddle_samples/PaddleNLP/ernie-3.0-nano-zh`
- raw data: `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json`

Results:

| Stage | Result |
| --- | --- |
| Raw official import | failed due to missing `graph_net.paddle.backend` path |
| Minimal backend path compatibility patch | import passed |
| Direct dygraph forward | passed on `Place(gpu:0)`, output `[1, 312]` |
| Official `compiler=nope` benchmark | process completed but benchmark status `eager:failed compiled:failed` |
| Failure locus | `paddle.jit.to_static` / dy2static path around generated `_C_ops.full` call |

A separate `_C_ops` signature probe shows `_C_ops.full(..., paddle.int64, gpu_place)` succeeds directly, so the failure is more likely a generated-code + dy2static compatibility issue than a simple missing backend kernel.

## Resume-Grade Project Direction

Recommended project framing:

> Paddle GraphNet-style backend and inference probe on MetaX C500

This should be positioned as a framework/runtime project, not an AI4Science app demo.

Suggested resume bullets after deeper completion:

- Built a reproducible PaddlePaddle backend probe on MetaX C500, identifying that the vendor image exposes the accelerator through Paddle's CUDA-compatible `gpu:0` path rather than custom device dispatch.
- Validated Paddle Inference availability and core operator coverage on C500, including fp32/fp16 matmul, conv2d, layernorm, softmax, gather, and scatter.
- Connected PaddlePaddle GraphNet-style computation graph workloads to backend/kernel coverage by profiling dense, sparse-ish, mixed, and readout graph blocks relevant to tensor compiler benchmarks.
- Verified `paddle.jit.save/load` static graph execution for dense and mixed GraphNet-style blocks on C500.
- Brought up a real official GraphNet Paddle sample in direct dygraph mode and isolated the official static benchmark failure to a dy2static compatibility issue.
- Exported fixed-shape graph blocks to Paddle Inference and validated `PaddleInferPredictor` execution on C500.
- Identified a key compiler limitation in the current image: CINN is not compiled in, so this image is suitable for runtime/operator/inference analysis but not direct CINN-vs-baseline compiler benchmarking.

## How This Relates to Inference

This Paddle path complements the vLLM path:

| vLLM side | Paddle side |
| --- | --- |
| request scheduler | graph/runtime execution |
| KV cache | tensor storage and op memory traffic |
| TTFT / TPOT | op latency / static inference latency |
| CUDA Graph / compile path | static graph / Paddle Inference / CINN boundary |
| prefix cache workload | graph-level common substructure / reusable prefix analogy |

For interviews, the unifying story is:

> I investigated AI infra at two levels: LLM serving behavior with vLLM, and framework/backend behavior with PaddlePaddle. On C500, I measured serving-level TTFT/TPOT/KV-cache effects, then switched to Paddle to inspect backend dispatch, operator coverage, Paddle Inference availability, and GraphNet-style tensor compiler workloads.

## Next Work Items

1. Clone/read PaddlePaddle/GraphNet and summarize its graph format and benchmark pipeline.
2. Run one minimal GraphNet benchmark path if dependencies fit the C500 Paddle image.
3. Add repeated graph-size sweeps and `mx-smi` sampling around Paddle op groups.
4. Try to run Paddle Inference predictor under larger graph sizes and repeated request patterns.
5. Try to find or build a Paddle image with CINN enabled; current image reports `is_compiled_with_cinn() = False`.

## Concrete Project Plan

Working title:

> Paddle GraphNet Compiler-Workload Probe on MetaX C500

Phase 1: backend capability map, already started here.

- Detect Paddle package, backend route, device strings, custom-device availability, CINN availability, Paddle Inference API availability.
- Run core op compatibility smoke on C500 `gpu:0`.
- Classify the image as runtime/operator/inference capable, but not CINN-enabled.

Phase 2: mini GraphNet-style graph replay harness, completed as first pass.

- Build several synthetic graph workloads using Paddle ops:
  - dense block: matmul -> bias -> activation -> layernorm
  - conv block: conv2d -> activation -> normalization
  - sparse-ish block: gather -> elementwise -> scatter
  - mixed block: dense + gather/scatter + reduction
- Run dynamic graph latency and throughput.
- Convert supported graphs through `paddle.jit.save` and test static/Paddle Inference execution.
- Record unsupported ops, fallback behavior, and export failures.

Phase 3: GraphNet alignment.

- Read PaddlePaddle/GraphNet graph format and benchmark scripts.
- Map GraphNet graph operator categories to the mini harness categories.
- If dependencies are manageable, run one real GraphNet sample.
- If not, clearly document why the current C500 Paddle image is insufficient and use the mini harness as a controlled approximation.

Phase 4: resume-ready summary.

- Produce tables for backend capability, op coverage, dynamic vs static inference, and sparse-ish op latency.
- Write a short project README suitable for linking from a resume.
- Keep caveats explicit: C500 Paddle image is CUDA-compatible, CINN is absent, and current numbers are smoke-level until repeated runs are added.

## Sources

- PaddlePaddle/GraphNet repository: `https://github.com/PaddlePaddle/GraphNet`
- GraphNet technical report / arXiv entry linked by the repository: `https://arxiv.org/abs/2510.24035`

## Evidence Quality

Current evidence is enough for a first report and a resume-ready mini project, but not yet enough to claim full GraphNet benchmark completion.

Completed:

- Paddle backend/device probe
- Paddle Inference API availability check
- core op smoke on `gpu:0`
- sparse-ish gather/scatter op smoke
- mini GraphNet-style dynamic graph workload
- `paddle.jit.save/load` static graph smoke for dense and mixed blocks
- Paddle Inference predictor smoke for dense and mixed graph blocks

Not completed yet:

- broader official GraphNet static benchmark replay across multiple samples
- CINN benchmark comparison
- full deployment benchmark with repeated request patterns
- repeated runs across multiple graph sizes


## Official Static Patch Follow-Up

A temporary generated-code patch experiment is recorded in `raw/metax-c500-paddle/official_graphnet_static_patch/official_graphnet_static_patch_probe.json` and implemented by `scripts/official_graphnet_static_patch_probe.py`. Rewriting all 159 generated `_C_ops` calls in the official `ernie-3.0-nano-zh` sample to higher-level Paddle APIs allowed the official `compiler=nope` benchmark to complete with `eager:success compiled:success`. The measured e2e median was 4.415 ms for eager and 4.409 ms for compiled; GPU event timing reported 0.0 ms, so GPU-only timing is not reliable in this Paddle/MACA image.


## Official GraphNet Timing After Generated-Code Rewrite

`scripts/official_graphnet_static_patch_probe.py` now rewrites all 159 generated `_C_ops` calls in the official `ernie-3.0-nano-zh` sample to high-level Paddle APIs in a temporary workdir. The official `compiler=nope` benchmark then succeeds on C500 with `eager:success compiled:success`.

| Mode | e2e median ms | e2e mean ms | Status |
| --- | ---: | ---: | --- |
| eager | 4.415 | 4.450 | success |
| compiled/nope | 4.409 | 4.411 | success |

The GPU-only event timing reports 0.0 ms, so it is treated as unsupported/unreliable for this image.
