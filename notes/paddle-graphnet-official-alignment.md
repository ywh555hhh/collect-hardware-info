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
- official GraphNet `ernie-3.0-nano-zh` direct dygraph forward on C500
- official GraphNet benchmark static-path failure diagnosis
- low-level `_C_ops.full` / `_C_ops.cast` signature probe
- generated `_C_ops` rewrite pass for official GraphNet Paddle samples
- five-sample official PaddleNLP GraphNet sweep with final 5/5 pass rate

Current limitation:

- `paddle.is_compiled_with_cinn()` is `False`, so this image cannot support a real CINN-vs-eager compiler speedup claim.

Current usable project framing:

> Paddle GraphNet-style backend and inference probe on MetaX C500.

This is a runtime/backend readiness project with selected official GraphNet Paddle sample timing, not a full GraphNet/CINN speedup benchmark yet.


## Official Sample Bring-Up Result

Raw data:

- `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json`
- `raw/metax-c500-paddle/official_graphnet/official_graphnet_ernie_nano_nope.stderr.log`
- `raw/metax-c500-paddle/cops_signature/paddle_cops_signature_probe.json`

Script:

- `scripts/official_graphnet_c500_bringup_probe.py`
- `scripts/paddle_cops_signature_probe.py`

Tested official sample:

- `paddle_samples/PaddleNLP/ernie-3.0-nano-zh`

Results:

| Stage | Result | Evidence |
| --- | --- | --- |
| Raw official import | failed | `ModuleNotFoundError: No module named 'graph_net.paddle.backend'` |
| Minimal backend path patch | passed | copied `graph_net_bench/paddle/backend/*.py` to `graph_net/paddle/backend/` in temporary workdir |
| Direct dygraph forward | passed | output shape `[1, 312]`, dtype `paddle.float32`, place `Place(gpu:0)` |
| Official `compiler=nope` benchmark | process completed, benchmark status failed | `eager:failed compiled:failed` |
| Hardware detection | passed | benchmark reports hardware `MetaX C500` |

The official benchmark failure is specific:

```text
ValueError: full: argument (position 3) must be one of paddle::DataType, but got paddle.base.libpaddle.VarType
```

It occurs inside the benchmark's static conversion path around generated sample code:

```python
full_0 = paddle._C_ops.full(
    [], float("0"), paddle.int64, paddle.framework._current_expected_place()
)
```

A separate low-level signature probe shows `_C_ops.full([], 0.0, paddle.int64, gpu_place)` succeeds directly on this image. Therefore the current evidence suggests the failure is tied to GraphNet generated code under `paddle.jit.to_static` / dy2static transformation on Paddle 2.6.0+MACA, not a simple missing C500 kernel.

This became the next engineering step: make official GraphNet static benchmark replay compatible with the vendor Paddle image, then compare static/nope/CINN if a CINN-enabled image becomes available.

## Compatibility Harness Implemented

The implemented path is a thin compatibility harness rather than a broad rewrite:

1. Unpack official GraphNet into a temporary workdir.
2. Keep the backend import compatibility patch minimal.
3. Run samples on Paddle's accepted `gpu:0` device path.
4. Rewrite generated low-level `paddle._C_ops.*` calls to high-level Paddle APIs in the temporary workdir.
5. Re-run official `graph_net_bench.paddle.test_compiler --compiler nope`.
6. Aggregate status, patch count, e2e timing, and failure lines across samples.

Current deliverables:

- compatibility patch notes
- one-sample and five-sample official GraphNet timing logs
- operator/device coverage matrix
- comparison with mini GraphNet-style probe results

## Resume Interpretation

A strong but honest resume claim after the current work:

> Built a PaddlePaddle/GraphNet compatibility harness on MetaX C500, identifying device API compatibility gaps (`cuda` rejected, `gpu:0` accepted), validating dense/sparse/static/Paddle Inference graph execution, rewriting generated `_C_ops` calls to high-level Paddle APIs, and running official `compiler=nope` timing across five PaddleNLP GraphNet samples.

Do not claim yet:

- full 2.7K+ GraphNet corpus benchmark completion
- CINN speedup on C500
- kernel-level profiling with vendor tools

The value is still substantial: it shows framework internals judgment, benchmark bring-up discipline, and inference/static-graph deployment awareness.


## Official Static Patch Follow-Up

A temporary generated-code patch experiment is recorded in `raw/metax-c500-paddle/official_graphnet_static_patch/official_graphnet_static_patch_probe.json` and implemented by `scripts/official_graphnet_static_patch_probe.py`. Rewriting all 159 generated `_C_ops` calls in the official `ernie-3.0-nano-zh` sample to higher-level Paddle APIs allowed the official `compiler=nope` benchmark to complete with `eager:success compiled:success`. The measured e2e median was 4.415 ms for eager and 4.409 ms for compiled; GPU event timing reported 0.0 ms, so GPU-only timing is not reliable in this Paddle/MACA image.


## Official Benchmark Timing After Rewrite

The final static patch experiment rewrites all 159 generated `_C_ops` calls in `ernie-3.0-nano-zh/model.py` to higher-level Paddle APIs inside a temporary workdir. With the backend import compatibility patch, the official `graph_net_bench.paddle.test_compiler --compiler nope` path completes successfully on C500.

| Mode | Status | e2e median ms | e2e mean ms | GPU event median ms |
| --- | --- | ---: | ---: | ---: |
| eager | success | 4.415 | 4.450 | 0.000 |
| compiled/nope | success | 4.409 | 4.411 | 0.000 |

Interpretation: this proves the official Paddle sample can be made to run through GraphNet benchmark infrastructure on C500. The near-identical eager and compiled/nope timings are expected because `nope` is not an optimizing compiler. GPU event timing is unusable in this image because it reports 0.0 ms, so e2e timing should be used until a reliable vendor/Paddle event timing path is found.

## Official Multi-Sample Sweep

The stronger follow-up is recorded in `raw/metax-c500-paddle/official_graphnet_multi_sample_final/official_graphnet_multi_sample_sweep.json` and summarized in `notes/paddle-graphnet-multi-sample-sweep.md`.

| Official GraphNet Paddle Sample | Generated `_C_ops` Rewritten | Eager e2e Median ms | Compiled/nope e2e Median ms | Status |
| --- | ---: | ---: | ---: | --- |
| `PaddleNLP/ernie-3.0-nano-zh` | 159 | 4.483 | 4.415 | pass |
| `PaddleNLP/ernie-3.0-tiny-pico-v2-zh` | 122 | 3.442 | 3.356 | pass |
| `PaddleNLP/ernie-3.0-tiny-base-v2-zh` | 419 | 12.377 | 12.438 | pass |
| `PaddleNLP/rocketqa-nano-cross-encoder` | 159 | 4.585 | 4.590 | pass |
| `PaddleNLP/uer_chinese-roberta-tiny` | 90 | 3.171 | 3.120 | pass |

The first five-sample sweep passed 3/5. After correcting the sample path and adding `_C_ops.full_like` conversion, the final sweep passed 5/5 and rewrote 949 generated calls. This is the best current evidence that the harness is repeatable across selected official PaddleNLP GraphNet samples, not just a one-sample demo.
