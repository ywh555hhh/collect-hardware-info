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
- `raw/metax-c500-paddle/mini_graphnet/mini_graphnet_probe.json`

Scripts:

- `scripts/paddle_backend_probe.py`
- `scripts/paddle_op_probe.py`
- `scripts/paddle_mini_graphnet_probe.py`

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
| Paddle Inference API | available |
| `paddle.utils.run_check()` | pass |

Interpretation:

- C500 is exposed to Paddle as a CUDA/GPU-compatible backend.
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

## Resume-Grade Project Direction

Recommended project framing:

> Paddle GraphNet-style backend and inference probe on MetaX C500

This should be positioned as a framework/runtime project, not an AI4Science app demo.

Suggested resume bullets after deeper completion:

- Built a reproducible PaddlePaddle backend probe on MetaX C500, identifying that the vendor image exposes the accelerator through Paddle's CUDA-compatible `gpu:0` path rather than custom device dispatch.
- Validated Paddle Inference availability and core operator coverage on C500, including fp32/fp16 matmul, conv2d, layernorm, softmax, gather, and scatter.
- Connected PaddlePaddle GraphNet-style computation graph workloads to backend/kernel coverage by profiling dense, sparse-ish, mixed, and readout graph blocks relevant to tensor compiler benchmarks.
- Verified `paddle.jit.save/load` static graph execution for dense and mixed GraphNet-style blocks on C500.
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
3. Try Paddle Inference predictor execution from the saved `.pdmodel` artifacts.
4. Add repeated graph-size sweeps and `mx-smi` sampling around Paddle op groups.
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

Not completed yet:

- full GraphNet workload replay
- CINN benchmark comparison
- Paddle Inference predictor latency comparison
- repeated runs across multiple graph sizes
