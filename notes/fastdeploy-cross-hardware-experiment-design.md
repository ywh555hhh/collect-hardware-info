# FastDeploy LLM serving cross-hardware experiment design

Date: 2026-08-13

This document turns the Paddle/FastDeploy direction into an inference-framework optimization project. The hardware scope is:

- NVIDIA RTX 4090 / 24GB
- MetaX C500 / current C500 sGPU slice first
- Iluvatar MR-V100 / 32GB

Only C500 is executed in the current phase. RTX 4090 and MR-V100 are included as the target comparison matrix so that the C500 work does not become an isolated demo.

## Project Positioning

Working title:

> FastDeploy LLM Serving on Heterogeneous GPUs: request lifecycle, KV cache, prefix cache, and scheduler profiling

The goal is not merely to run a Paddle model. The goal is to inspect the same FastDeploy serving stack across three accelerator families and explain which performance mechanisms survive across backends:

- request lifecycle: model load -> KV cache allocation -> warmup -> prefill -> decode -> streaming output
- scheduling: max concurrent sequences, max batched prefill tokens, chunked prefill, mixed prefill/decode traffic
- memory system: block size, GPU memory utilization, total KV blocks, maximum cacheable tokens
- caching: prefix caching hit/miss behavior under mixed shared-prefix workloads
- graph/kernel path: CUDA Graph / graph optimization where supported, backend-specific attention and sampling paths
- serving metrics: TTFT, TPOT, ITL, end-to-end latency, output throughput, request throughput, p50/p90/p99

This is the Paddle ecosystem equivalent of a vLLM/SGLang serving investigation.

## Why FastDeploy

FastDeploy exposes the right inference-engine concepts:

- OpenAI-compatible API server: `fastdeploy.entrypoints.openai.api_server`
- offline `LLM.generate` API
- benchmark client: `fastdeploy.benchmarks.serve`
- KV cache parameters: `block_size`, `gpu_memory_utilization`, `num_gpu_blocks_override`, `kv_cache_ratio`
- scheduler parameters: `max_num_seqs`, `max_num_batched_tokens`, `max_num_partial_prefills`
- prefix cache: `enable_prefix_caching`, optional CPU `swap_space`
- chunked prefill: `enable_chunked_prefill`
- graph optimization: `graph_optimization_config`
- backend-specific worker paths, including Metax and Iluvatar code paths in the repository

Upstream references inspected locally from a fresh FastDeploy clone:

- `docs/zh/get_started/installation/nvidia_gpu.md`
- `docs/zh/get_started/installation/metax_gpu.md`
- `docs/zh/get_started/installation/iluvatar_gpu.md`
- `docs/zh/parameters.md`
- `docs/zh/features/prefix_caching.md`
- `docs/zh/features/chunked_prefill.md`
- `fastdeploy/benchmarks/serve.py`
- `fastdeploy/entrypoints/openai/api_server.py`

## Hardware Roles

| Hardware | Role | Expected FastDeploy Path | Key Question |
| --- | --- | --- | --- |
| RTX 4090 24GB | NVIDIA baseline | `fastdeploy-gpu` + CUDA 12.6/12.9, SM89 | What does the same serving workload look like on the best-supported path? |
| MetaX C500 | MetaX compatibility target | MetaX/MACA Paddle route; current image is not FastDeploy-ready | Can a C500 image run the C550-oriented MetaX FastDeploy stack, and where does it fail first? |
| Iluvatar MR-V100 32GB | non-NVIDIA serving comparison | `paddlepaddle-iluvatar` + `fastdeploy_iluvatar_gpu` | How do block size, CUDAGraph, sampling, and KV capacity differ from NVIDIA? |

## Model Ladder

The model ladder should be small-to-large, with each step answering a different question:

| Level | Model | Purpose | Hardware Fit |
| --- | --- | --- | --- |
| L0 | import-only / tokenizer-only | Verify Python stack and API surface | all |
| L1 | `baidu/ERNIE-4.5-0.3B-Paddle` | FastDeploy minimal LLM serving workload | 4090 first; C500/MR-V100 if image supports |
| L2 | small Qwen/ERNIE Paddle-format model if available locally | Compare against existing vLLM Qwen3 probes | 4090/C500 |
| L3 | `ERNIE-4.5-21B-A3B-Paddle` WINT8/WINT4 | realistic FastDeploy official workload | MR-V100 and larger-memory hardware; maybe 4090 if quantized and memory fits |

Use L1 for the first resume-grade inference story. Larger models are useful only after the serving metrics pipeline is solid.

## Experiment Matrix

### Stage 0: Readiness Gate

Purpose: prove whether the host can run FastDeploy before spending time on model downloads.

Script:

- `scripts/fastdeploy_readiness_probe.py`

Required checks:

- accelerator visibility: `nvidia-smi`, `mx-smi`, or `ixsmi`
- Paddle package and version
- Paddle device string and custom device types
- FastDeploy module import
- OpenAI server dependencies: `fastapi`, `uvicorn`, `openai`
- FastDeploy verification imports:
  - `from paddle.jit.marker import unified`
  - `from fastdeploy.model_executor.ops.gpu import beam_search_softmax`

Output:

- `raw/<hardware>-fastdeploy/fastdeploy_<hardware>_readiness_probe.json`

### Stage 1: Offline LLM Smoke

Purpose: verify `LLM.generate` works before server overhead enters the picture.

Workload:

- prompts: 2 short prompts, then 8 mixed prompts
- output length: 64, 256
- sampling: greedy and default sampling

Metrics:

- model load time
- worker launch time
- total generation time
- generated tokens
- any backend selection lines from logs

Expected report:

- "Does the FastDeploy engine start?"
- "Which attention/sampling/backend path does it choose?"
- "How much KV cache capacity is detected?"

### Stage 2: OpenAI-Compatible Server Smoke

Purpose: confirm the actual serving path.

Server template:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model <MODEL> \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85
```

Client:

- FastDeploy's `fastdeploy.benchmarks.serve`
- existing OpenAI-compatible benchmark harnesses in this repo can be reused because FastDeploy exposes `/v1/chat/completions`

Metrics:

- health check result
- first successful non-streaming response
- first successful streaming response
- server log markers for cache blocks and backend selection

### Stage 3: Prefill / Decode Shape Sweep

Purpose: show real inference-framework reasoning.

Variables:

| Axis | Values |
| --- | --- |
| input tokens | 128, 512, 2048, 4096 |
| output tokens | 32, 128, 512 |
| concurrency | 1, 4, 8, 16, 32 |

Metrics:

- TTFT
- TPOT / ITL
- end-to-end latency
- request throughput
- output token throughput
- p50/p90/p99
- GPU memory before/after startup

Interpretation target:

- longer input mainly stretches prefill / TTFT
- longer output stresses decode / TPOT
- higher concurrency trades throughput against tail latency

### Stage 4: KV Cache Capacity Frontier

Purpose: quantify memory-management limits.

Variables:

| Axis | Values |
| --- | --- |
| `max_model_len` | 2048, 4096, 8192, 16384, 32768 |
| `gpu_memory_utilization` | 0.70, 0.80, 0.90 |
| `max_num_seqs` | 8, 16, 32, 64, 128 |
| `block_size` | backend default; MR-V100 also test 16 because upstream Iluvatar docs use 16 |

Extract from logs:

- total GPU KV blocks
- prefill/decode KV split if logged
- maximum serviceable concurrency
- OOM boundary

Interpretation target:

- how each backend converts memory budget into cache blocks
- how much context length reduces concurrency
- whether the backend needs a different block size tuning

### Stage 5: Prefix Cache Mixed Workload

Purpose: make the experiment look like a serving optimization project, not a throughput demo.

Workloads:

| Workload | Shared Prefix Ratio | Description |
| --- | ---: | --- |
| unique | 0% | every request has unrelated context |
| mixed-low | 25% | some shared system prompt / retrieved context |
| mixed-mid | 50% | half the traffic shares prefix |
| mixed-high | 75% | mostly shared prefix |
| all-shared | 100% | best-case cache reuse |

Variables:

- `enable_prefix_caching`: off/on
- `swap_space`: off/on where memory allows
- prompt length: 512, 2048
- concurrency: 8, 16, 32

Metrics:

- TTFT improvement
- request throughput
- p90/p99 tail latency
- cache hit/miss evidence from logs or metrics

Interpretation target:

- prefix cache primarily reduces repeated prefill work
- mixed workloads can improve mean latency while still exposing tail effects

### Stage 6: Chunked Prefill Mixed Traffic

Purpose: test scheduler behavior under long prompts plus active decode traffic.

Workload:

- 70% short prompts: 128-512 input tokens, 64 output tokens
- 30% long prompts: 4096-8192 input tokens, 128 output tokens

Variables:

- `enable_chunked_prefill`: off/on
- `max_num_batched_tokens`: 512, 1024, 2048, 4096, 8192
- `max_num_partial_prefills`: 1, 2, 4

Metrics:

- short-request TTFT p90/p99
- long-request completion latency
- aggregate output throughput
- OOM / scheduler rejection rate

Interpretation target:

- chunked prefill should protect decode responsiveness and reduce memory spikes
- smaller chunk budgets improve inter-token responsiveness but can hurt first-token latency

### Stage 7: Graph Optimization / CUDAGraph

Purpose: isolate launch-overhead mitigation where supported.

Variables:

- `graph_optimization_config={"use_cudagraph": true, "graph_opt_level": 0}`
- `graph_optimization_config={"use_cudagraph": false, "graph_opt_level": 0}`
- fixed decode-heavy workload: input 128, output 512, concurrency 1/4/16

Metrics:

- TPOT
- ITL
- memory overhead
- startup/warmup overhead

Interpretation target:

- NVIDIA 4090 is expected to be the cleanest CUDA Graph baseline
- Iluvatar docs expose CUDAGraph usage in examples, so MR-V100 should be tested
- C500 support must be treated as unknown until the correct MetaX image is available

## Report Set

The final project should have one summary and one per-hardware report:

- `notes/fastdeploy-cross-hardware-experiment-design.md`
- `notes/fastdeploy-c500-readiness-report.md`
- `notes/fastdeploy-4090-serving-report.md`
- `notes/fastdeploy-c500-serving-report.md`
- `notes/fastdeploy-mrv100-serving-report.md`
- `notes/fastdeploy-cross-hardware-comparison.md`

Current phase only has the design and C500 readiness report.

## Resume Output Target

Final resume claim after the serving stages are actually run:

> Built a FastDeploy LLM serving benchmark harness across NVIDIA 4090, MetaX C500, and Iluvatar MR-V100; profiled TTFT/TPOT, KV-cache capacity, prefix-cache effectiveness, chunked-prefill scheduling, and graph-optimization trade-offs under mixed prompt workloads; identified backend-specific readiness and tuning gaps across Paddle-supported accelerator paths.

Current honest claim after C500 readiness only:

> Designed a FastDeploy LLM serving experiment suite for NVIDIA, MetaX, and Iluvatar GPUs and ran the C500 readiness gate, identifying that the current C500 Paddle 2.6/MACA image lacks FastDeploy, Paddle LLM serving dependencies, and the official MetaX FastDeploy verification surface.

