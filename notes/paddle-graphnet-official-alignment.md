# PaddlePaddle/GraphNet official alignment on MetaX C500

Date: 2026-08-12

This note aligns the official PaddlePaddle/GraphNet benchmark path with the C500 Paddle image. The goal is to turn the current mini GraphNet-style probes into a real GraphNet-oriented backend/runtime project.

## What GraphNet Is

GraphNet is a computation graph dataset and benchmark for tensor compiler research, not a conventional graph neural network library.

The official repository frames GraphNet around:

- 2.7K+ computation graphs from real deep learning models
- compiler evaluation across PaddlePaddle/CINN and PyTorch/TorchInductor
- correctness plus performance metrics such as Speedup Score and Error-aware Speedup Score
- hardware regression testing against reference outputs and timing logs

That makes GraphNet a strong fit for a Paddle/runtime/compiler resume story.

## Official Paddle Benchmark Surface

Important files in the official repository:

| Area | File | Why it matters |
| --- | --- | --- |
| Paddle compiler benchmark | `graph_net_bench/paddle/test_compiler.py` | Runs eager baseline and compiled backend, then reports correctness/performance |
| Paddle target-device regression | `graph_net_bench/paddle/test_target_device.py` | Replays models against reference logs/outputs on target hardware |
| Hardware test guide | `docs/hardware_test.md` | Describes reference generation and target-device regression workflow |
| Paddle sample runtime | `graph_net/paddle/run_model.py` | Loads/replays Paddle graph samples |
| Paddle sample utilities | `graph_net/paddle/utils.py` | Reconstructs tensors and sample metadata |
| Sample database | `db_samples/small10_torch_samples.db` | Small sample data included in the repo |

The official Paddle benchmark currently has two important assumptions for our C500 environment:

1. `graph_net_bench/paddle/test_compiler.py` asserts `args.device in ["cuda", "dcu", "xpu", "cpu"]`.
2. The C500 Paddle image accepts Paddle device strings `gpu` and `gpu:0`, but rejects `cuda` and `cuda:0`.

This means the official Paddle path will likely need a small device-string compatibility patch or wrapper before it can run on this C500 image.

## C500 Paddle Device Alias Probe

Raw data:

- `raw/metax-c500-paddle/device_alias/paddle_device_alias_probe.json`

Script:

- `scripts/paddle_device_alias_probe.py`

Observed results:

| Device string | Result | Current device / place |
| --- | --- | --- |
| `cuda` | fail | Paddle rejects the string |
| `cuda:0` | fail | Paddle rejects the string |
| `gpu` | pass | `gpu:0` / `Place(gpu:0)` |
| `gpu:0` | pass | `gpu:0` / `Place(gpu:0)` |
| `maca` | fail | Paddle rejects the string |
| `maca:0` | fail | Paddle rejects the string |
| `cpu` | pass | `cpu` / `Place(cpu)` |

Interpretation:

- The backend is CUDA-compatible at the compiled/runtime layer, but the Paddle Python device API wants `gpu` / `gpu:0`.
- Official GraphNet's `--device cuda` convention is not directly compatible with this image.
- A minimal compatibility layer should map GraphNet's logical `cuda` path to Paddle's `gpu:0` device string while preserving "GPU device" timing behavior.

## Current C500 Project State

Already completed in this repo:

- Paddle backend/device probe
- Paddle core op smoke on C500
- mini GraphNet-style dynamic graph workload
- `paddle.jit.save/load` static graph workload
- Paddle Inference `PaddleInferPredictor` workload
- device alias compatibility probe

Current limitation:

- `paddle.is_compiled_with_cinn()` is `False`, so this image cannot support a real CINN-vs-eager compiler speedup claim.

Current usable project framing:

> Paddle GraphNet-style backend and inference probe on MetaX C500.

This is a runtime/backend readiness project, not a full official GraphNet/CINN benchmark yet.

## Recommended Next Implementation

The highest-return next step is a thin compatibility harness rather than a broad rewrite:

1. Clone official GraphNet on the C500 machine.
2. Install only the minimum dependencies needed to import and run a small sample.
3. Patch or wrap the Paddle benchmark so logical GraphNet `cuda` maps to Paddle `gpu:0`.
4. Try `compiler=nope` first to validate eager/static replay without CINN.
5. Use C500 as target device against a tiny reference run, then record failure/success by sample.
6. Only after that, explore whether a CINN-enabled Paddle image exists.

Expected deliverables:

- compatibility patch notes
- one small official GraphNet sample run log or an explicit incompatibility report
- operator/device coverage matrix
- comparison with mini GraphNet-style probe results

## Resume Interpretation

A strong but honest resume claim after the current work:

> Built a PaddlePaddle backend-readiness and GraphNet-style inference probe on MetaX C500, identifying device API compatibility gaps (`cuda` rejected, `gpu:0` accepted), validating dense/sparse/static/Paddle Inference graph execution, and scoping the changes required to run official GraphNet hardware regression workloads on a non-NVIDIA accelerator.

Do not claim yet:

- full official GraphNet benchmark completion
- CINN speedup on C500
- kernel-level profiling with vendor tools

The value is still substantial: it shows framework internals judgment, benchmark bring-up discipline, and inference/static-graph deployment awareness.
