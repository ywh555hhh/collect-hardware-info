# FastDeploy C500 readiness report

Date: 2026-08-13

This is the first executed FastDeploy-related experiment on the current MetaX C500 machine. It is a readiness gate, not yet a serving benchmark. The goal is to decide whether the current C500 image can run FastDeploy LLM serving experiments, and if not, identify the first blocking layer precisely.

## Raw Data

- `raw/metax-c500-fastdeploy/fastdeploy_c500_readiness_probe.json`
- `raw/metax-c500-fastdeploy/fastdeploy_package_index_probe.log`

Script:

- `scripts/fastdeploy_readiness_probe.py`

## Hardware Visibility

The C500 device is visible to the host and idle at probe time.

| Item | Value |
| --- | --- |
| GPU | MetaX C500 |
| Driver / KMD | 3.8.30 |
| MACA | 3.0.0.8 |
| `mx-smi` | 2.2.6 |
| Instance shape | sGPU slice |
| sGPU quota | 25% compute / 16GB VRAM quota |
| Probe-time GPU util | 0% |
| Container status | current shell is already inside a Docker container |
| Docker CLI inside current container | not installed |
| Podman CLI inside current container | not installed |

This means the first blocker is not hardware visibility. The card is visible; the FastDeploy serving stack is missing or mismatched. The current shell is already inside a provider-managed Docker container, and there is no Docker/Podman CLI available inside it, so the official MetaX Docker image cannot be pulled and run from this current container directly.

## Paddle Runtime

| Check | Result |
| --- | --- |
| Python | 3.10.10 via `/opt/conda/bin/python3` |
| Paddle package | `paddlepaddle-gpu 2.6.0+maca3.0.0.5` |
| `paddle.__version__` | `2.6.0` |
| Paddle device | `gpu:0` |
| `is_compiled_with_cuda` | `True` |
| `is_compiled_with_xpu` | `False` |
| `is_compiled_with_cinn` | `False` |
| custom device types | `[]` |

Interpretation: this is the same Paddle 2.6 CUDA-compatible `gpu:0` route used in the earlier GraphNet work. It is useful for Paddle runtime and operator probes, but it is not the FastDeploy MetaX software route described by the current upstream FastDeploy docs.

## Python Serving Stack

Module availability from the readiness probe:

| Module | Found |
| --- | --- |
| `paddle` | `True` |
| `fastdeploy` | `False` |
| `paddlenlp` | `False` |
| `paddleformers` | `False` |
| `fastapi` | `False` |
| `uvicorn` | `False` |
| `openai` | `False` |
| `transformers` | `False` |

Installed package subset:

| Package | Version |
| --- | --- |
| `paddlepaddle-gpu` | `2.6.0+maca3.0.0.5` |


Readiness gaps reported by the probe:

- fastdeploy Python module is not installed
- Paddle LLM model/tokenizer stack is not installed
- OpenAI-compatible API server dependencies are missing
- Paddle was built without CINN; compiler speedup experiments are out of scope
- paddle.jit.marker.unified check failed; official FastDeploy MetaX verification may not pass
- FastDeploy custom GPU operator import check failed

## FastDeploy Verification Imports

| Import check | Result |
| --- | --- |
| `paddle_jit_marker_unified` | `fail: ModuleNotFoundError("No module named 'paddle.jit.marker'")` |
| `fastdeploy` | `fail: ModuleNotFoundError("No module named 'fastdeploy'")` |
| `fastdeploy_llm` | `fail: ModuleNotFoundError("No module named 'fastdeploy'")` |
| `fastdeploy_sampling_params` | `fail: ModuleNotFoundError("No module named 'fastdeploy'")` |
| `fastdeploy_openai_api_server` | `fail: ModuleNotFoundError("No module named 'fastdeploy'")` |
| `fastdeploy_gpu_ops` | `fail: ModuleNotFoundError("No module named 'fastdeploy'")` |


These failures are expected on this image because FastDeploy is not installed. The more important compatibility clue is `paddle.jit.marker.unified` missing, because the upstream FastDeploy MetaX installation verification imports it after installing the expected Paddle/FastDeploy stack.

## Package Index Probe

A non-mutating `pip index versions` check found:

| Package query | Result |
| --- | --- |
| `fastdeploy-gpu` from Paddle CUDA 12.6 stable index | available: `2.5.0` |
| `fastdeploy-metax-gpu` from Paddle MACA nightly index | no matching distribution |
| plain `fastdeploy` from Tsinghua PyPI mirror | available, latest seen `3.1.1` |
| `paddle-metax-gpu` from Paddle MACA nightly index | no matching distribution |

Interpretation:

- The visible `fastdeploy-gpu` wheel is the NVIDIA CUDA path, not the MetaX C500 path.
- The plain PyPI `fastdeploy` package is not enough evidence that FastDeploy LLM custom GPU ops will work on C500.
- For MetaX, upstream FastDeploy docs currently describe a C550-oriented source-build / `paddle-metax-gpu` route, while this rented C500 image provides `paddlepaddle-gpu 2.6.0+maca3.0.0.5`.

## Decision

Current C500 image status:

> Not FastDeploy-ready for LLM serving experiments.

This is a useful result because it prevents us from mislabeling a package problem as an inference-performance result. The current machine can continue to support Paddle runtime / GraphNet compatibility work, but FastDeploy serving experiments need a different C500/MetaX image or a source-build path that matches C500, MACA, and the available Paddle package.

## Required Next Environment

For the next C500 attempt, request one of these environment changes first:

1. switch the platform image to the official FastDeploy MetaX image or an equivalent provider image; or
2. get host-level Docker access / a VM image where Docker is available; or
3. build FastDeploy in-place only if the Paddle/MACA/MetaX package versions match upstream requirements.

Then ensure the runtime has:

- Python 3.10
- current Paddle custom-device / MetaX package expected by FastDeploy, preferably `paddle-metax-gpu` if available in the provider image
- FastDeploy source tree or built wheel
- `fastapi`, `uvicorn`, `openai`, `paddleformers`, `transformers`, `safetensors`
- successful verification imports:
  - `from paddle.jit.marker import unified`
  - `from fastdeploy import LLM, SamplingParams`
  - `from fastdeploy.model_executor.ops.gpu import beam_search_softmax`
- a small Paddle-format LLM, ideally `baidu/ERNIE-4.5-0.3B-Paddle`, cached locally

## Next FastDeploy Serving Experiments Once Ready

After the readiness gate passes, run the stages from `notes/fastdeploy-cross-hardware-experiment-design.md` in this order:

1. offline `LLM.generate` smoke
2. OpenAI-compatible API server smoke
3. prefill/decode shape sweep
4. KV cache capacity frontier
5. prefix cache mixed workload
6. chunked prefill mixed traffic
7. graph optimization / CUDAGraph where supported

## Resume Interpretation

Safe current claim:

> Designed a FastDeploy LLM serving experiment suite for NVIDIA 4090, MetaX C500, and Iluvatar MR-V100, and executed the C500 readiness gate. The current C500 Paddle 2.6/MACA image exposes the accelerator through `gpu:0` but lacks FastDeploy, Paddle LLM serving dependencies, and the upstream MetaX FastDeploy verification surface, so it should be treated as a runtime/GraphNet image rather than a FastDeploy serving image.

Do not claim yet:

- FastDeploy serving runs on this C500 image
- TTFT/TPOT data from FastDeploy on C500
- prefix-cache or chunked-prefill speedups on C500
- C500/C550 FastDeploy compatibility without a matching image
