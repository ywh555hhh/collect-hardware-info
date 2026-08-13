# FastDeploy C500 hard-install attempt

Date: 2026-08-13

This note records an in-place FastDeploy hard-install attempt on the current MetaX C500 Paddle image. The purpose was to test whether the current provider image can be turned into a real FastDeploy LLM serving environment by installing dependencies one by one, after the official Docker-image route proved unavailable from inside the container.

## Raw Data

- `raw/metax-c500-fastdeploy/hardinstall/import_probe_after_hardinstall.json`
- `raw/metax-c500-fastdeploy/hardinstall/final_import_probe.json`
- `raw/metax-c500-fastdeploy/hardinstall/shim_import_probe.json`

## Environment

| Item | Value |
| --- | --- |
| Device | MetaX C500 sGPU slice |
| Compute / VRAM quota | 25% compute / 16GB VRAM quota |
| Python | 3.10.10 |
| Paddle | `paddlepaddle-gpu 2.6.0+maca3.0.0.5` |
| Paddle device route | CUDA-compatible `gpu:0` |
| MACA | 3.0.0.8 |
| Docker inside current shell | unavailable |
| Isolated venv | `/data/fd-hardinstall/venv` |
| FastDeploy source path | `/data/fd-hardinstall/FastDeploy` |

The install was intentionally isolated under `/data/fd-hardinstall` and used `PYTHONPATH=/data/fd-hardinstall/FastDeploy:$PYTHONPATH` instead of mutating the system Python environment.

## What Worked

The pure Python dependency layer can be pushed forward manually.

After installing the missing runtime packages, these pieces became importable:

| Component | Result |
| --- | --- |
| `fastapi` | OK |
| `uvicorn` | OK |
| `transformers` | OK |
| `paddleformers==0.4.1` | OK |
| `fastdeploy.engine.args_utils` | OK |
| `fastdeploy.cache_manager` | OK |
| `fastdeploy.model_executor.ops.gpu` package | OK as a Python package |

With a local compatibility shim for missing Paddle 3.x-style APIs, these top-level FastDeploy objects also imported:

| Import | Result |
| --- | --- |
| `import fastdeploy` | OK with shim |
| `from fastdeploy import LLM` | OK with shim |
| `from fastdeploy import SamplingParams` | OK with shim |

This means the current image can support a Python-level compatibility probe. It does not mean the FastDeploy serving runtime is operational.

## Blocking Layer

Without shims, FastDeploy currently fails at Paddle API compatibility:

| Import check | Error |
| --- | --- |
| `import fastdeploy` | `AttributeError("module 'paddle' has no attribute 'enable_compat'")` |
| `from fastdeploy import LLM` | `AttributeError("module 'paddle' has no attribute 'enable_compat'")` |
| `from fastdeploy import SamplingParams` | `AttributeError("module 'paddle' has no attribute 'enable_compat'")` |

FastDeploy also tries to load GPU custom ops and logs:

```text
decide_module error, load custom_ops from .fastdeploy_ops: module 'paddle.device' has no attribute 'get_device_properties'
```

After monkey-patching both `paddle.enable_compat` and `paddle.device.get_device_properties`, the Python package gets farther, but the GPU op surface is still absent:

| Import check | Error |
| --- | --- |
| `from fastdeploy.model_executor.ops.gpu import beam_search_softmax` | `ImportError("cannot import name 'beam_search_softmax' ...")` |

This is the decisive serving blocker. A FastDeploy LLM runtime needs compiled custom GPU ops for attention, sampling, cache management, MoE, and related kernels. A Python-only install is not enough.

## Build-Path Mismatch

The current image exposes the MetaX card through a CUDA-compatible Paddle route:

| Check | Result |
| --- | --- |
| `paddle.is_compiled_with_cuda()` | `True` |
| `paddle.device.get_all_custom_device_type()` | `[]` |
| `paddle.device.is_compiled_with_custom_device("metax_gpu")` | not available / false in this stack |
| `nvcc` | unavailable |
| `mxcc` | available |

This creates a structural mismatch for FastDeploy source builds:

- FastDeploy's normal build logic sees CUDA-compatible Paddle and tends toward the NVIDIA `gpu_ops` path.
- The current container does not have `nvcc`, so the NVIDIA CUDA custom ops cannot be built cleanly.
- FastDeploy has MetaX-specific custom ops under `custom_ops/metax_ops`, but that path expects a Paddle custom-device route such as `metax_gpu`.
- The current Paddle package exposes only `gpu:0`, not a `metax_gpu` custom device.
- FastDeploy `develop` expects newer Paddle APIs such as `paddle.enable_compat` and `paddle.device.get_device_properties`, while this image provides Paddle 2.6.

## Decision

Hard install is useful for diagnosis, but it does not turn this image into a real FastDeploy serving image.

Current status:

> Python dependency hard-install: partially successful.  
> FastDeploy Python import with shim: partially successful.  
> FastDeploy LLM serving runtime: not ready, blocked by Paddle API mismatch and missing compiled GPU custom ops.

The right next move depends on the goal:

| Goal | Recommended route |
| --- | --- |
| Collect compatibility evidence | keep this hard-install probe as-is |
| Run real TTFT / TPOT / prefix-cache / chunked-prefill data | switch to a provider image that already matches FastDeploy MetaX requirements |
| Do a deeper engineering project | port FastDeploy source to this Paddle 2.6 / MACA 3.0 / `gpu:0` environment by forcing the MetaX op build path and patching API gaps |

## Resume-Safe Claim

Safe claim:

> Attempted an isolated source hard-install of Paddle FastDeploy on a MetaX C500 Paddle 2.6/MACA image, installed the Python serving dependency layer, and localized the blocker to Paddle API drift plus missing compiled GPU custom ops. The current image can support Paddle runtime and compatibility studies, but not valid FastDeploy LLM serving benchmarks.

Do not claim:

- FastDeploy serving ran successfully on this C500 image.
- C500 FastDeploy TTFT / TPOT / prefix-cache data exists.
- The `beam_search_softmax` and related custom GPU ops are available.
- The current Paddle 2.6 `gpu:0` route is equivalent to the upstream FastDeploy MetaX custom-device route.
