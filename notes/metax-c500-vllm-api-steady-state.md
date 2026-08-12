# MetaX C500 vLLM steady-state API benchmark

采集时间：2026-08-12

这轮实验从“模型能不能跑”推进到“一个常驻 vLLM API server 在 warm 状态下怎么随并发、执行模式和 prompt 长度变化”。

目标不是做严格榜单，而是形成推理框架工程师面试里能讲清楚的材料：

- API server 生命周期：model load -> backend selection -> KV cache allocation -> warmup -> request serving
- eager vs non-eager：兼容路径 vs torch.compile + CUDAGraph 路径
- prompt length：prefill 变重后，对延迟和吞吐的影响
- hardware sampling：用 `mx-smi` 采 GPU/AP/HBM bandwidth，证明请求阶段确实在打设备

## Artifacts

| Experiment | Raw data |
| --- | --- |
| LFM2.5 eager, prompt ~=237 tokens | `raw/metax-c500-vllm/api_bench_lfm25_warm/lfm25_eager_prompt128.json` |
| LFM2.5 eager server log | `raw/metax-c500-vllm/api_bench_lfm25_warm/lfm25_eager_prompt128.server.log` |
| LFM2.5 non-eager, prompt ~=237 tokens | `raw/metax-c500-vllm/api_bench_lfm25_non_eager_warm/lfm25_noneager_prompt128.json` |
| LFM2.5 non-eager, prompt ~=757 tokens | `raw/metax-c500-vllm/api_bench_lfm25_non_eager_warm/lfm25_noneager_prompt512.json` |
| Harness | `scripts/vllm_api_bench.py` |

Notes:

- The script's `--prompt-token-len` is approximate. Actual prompt tokens came from vLLM API `usage.prompt_tokens`.
- Latency is end-to-end non-streaming request latency, not TTFT/TPOT split.
- Each concurrency group discards warmup requests before measurement.
- `mx-smi` samples are coarse and asynchronous; treat them as evidence, not a profiler replacement.

## Environment

| Item | Value |
| --- | --- |
| GPU | MetaX C500, current sGPU slice |
| Visible memory | ~15.6GB |
| MACA | 3.5.3.20 |
| vLLM | 0.17.0 |
| vllm_metax | 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8 |
| PyTorch | 2.8.0+metax3.5.3.9 |
| Model | LFM2.5-1.2B-Instruct |
| Serving mode | OpenAI-compatible `/v1/chat/completions` |
| `max_model_len` | 2048 |
| `gpu_memory_utilization` | 0.70 |
| `max_tokens` | 64 |

## Eager vs Non-Eager

Same model, same context limit, same measured request shape:

- actual prompt tokens: ~237
- measured requests per concurrency: 16
- warmup requests per concurrency: 4
- concurrency: 1, 2, 4, 8, 16

### Startup

| Mode | startup_s |
| --- | ---: |
| eager | 63.741 |
| non-eager | 68.516 |

The non-eager path was only ~4.8s slower in this warm-cache run. Earlier cold non-eager probing saw larger compile overhead, so the practical startup cost depends on compile cache state.

The non-eager server log confirms this path:

```text
compilation_config mode: VLLM_COMPILE
cudagraph_mode: FULL_AND_PIECEWISE
torch.compile takes 4.99 s in total
Graph capturing finished in 3 secs, took 0.15 GiB
```

### Request Results

| mode | concurrency | ok | p50 latency ms | p90 latency ms | completion tok/s wall | total tok/s wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 1 | 16 | 450.807 | 457.563 | 117.770 | 644.401 |
| eager | 2 | 16 | 500.317 | 580.266 | 217.074 | 1141.958 |
| eager | 4 | 16 | 495.850 | 532.474 | 425.894 | 2310.365 |
| eager | 8 | 16 | 546.738 | 582.915 | 743.162 | 3997.286 |
| eager | 16 | 16 | 816.839 | 821.259 | 1075.983 | 5660.394 |
| non-eager | 1 | 16 | 225.643 | 230.136 | 234.190 | 1281.417 |
| non-eager | 2 | 16 | 268.544 | 272.862 | 412.939 | 2178.291 |
| non-eager | 4 | 16 | 268.199 | 275.617 | 787.253 | 4307.612 |
| non-eager | 8 | 16 | 298.280 | 305.468 | 1404.785 | 7686.561 |
| non-eager | 16 | 16 | 358.870 | 369.228 | 2265.893 | 12327.092 |

Interpretation:

- Non-eager is much better in this serving shape: roughly 2x completion-token throughput at every concurrency level.
- The likely reason is torch.compile + CUDAGraph reducing per-step overhead and replaying a captured execution path.
- Eager remains valuable as a compatibility/debug path, but it is not the serving path to optimize around if non-eager is stable.
- At concurrency 16, non-eager still has low p90 relative to eager, which suggests the graph path is helping both throughput and tail latency in this small benchmark.

## Prompt Length Sweep

Same non-eager mode, same output cap, but prompt length changes:

- short prompt actual tokens: ~237
- longer prompt actual tokens: ~757

| prompt tokens | concurrency | ok | p50 latency ms | p90 latency ms | completion tok/s wall | total tok/s wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 237 | 1 | 16 | 225.643 | 230.136 | 234.190 | 1281.417 |
| 237 | 2 | 16 | 268.544 | 272.862 | 412.939 | 2178.291 |
| 237 | 4 | 16 | 268.199 | 275.617 | 787.253 | 4307.612 |
| 237 | 8 | 16 | 298.280 | 305.468 | 1404.785 | 7686.561 |
| 757 | 1 | 12 | 281.197 | 284.649 | 227.372 | 2916.759 |
| 757 | 2 | 12 | 322.000 | 326.176 | 396.241 | 5083.027 |
| 757 | 4 | 12 | 352.473 | 680.216 | 552.823 | 7091.684 |
| 757 | 8 | 12 | 423.804 | 444.789 | 980.912 | 12583.262 |

Interpretation:

- Longer prompts increase p50 latency because prefill work is larger before decode can proceed.
- Completion-token throughput drops at higher concurrency when prompt length grows: at concurrency 8 it falls from ~1405 tok/s to ~981 tok/s.
- Total token throughput rises because the benchmark counts prompt tokens too; this does not mean decode got faster. It means the server processed far more prefill tokens per request.
- The p90 spike at prompt ~=757 / concurrency 4 suggests batching/scheduling artifacts; this needs repeated runs before treating it as a stable tail-latency finding.

## Hardware Sampling

Each measured group optionally sampled:

- `mx-smi --show-usage`
- `mx-smi --show-ap-usage`
- `mx-smi --show-hbm-bandwidth`

Sample aggregates:

| run | concurrency | avg GPU util | max GPU util | avg HBM MB/s | max HBM MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager / 237 tok | 1 | 20.9% | 22% | 288286 | 299852 |
| eager / 237 tok | 8 | 22.0% | 24% | 219452 | 252727 |
| non-eager / 237 tok | 1 | 39.2% | 41% | 562914 | 594686 |
| non-eager / 237 tok | 8 | 37.0% | 37% | 481102 | 481102 |
| non-eager / 757 tok | 1 | 37.8% | 41% | 547867 | 602769 |
| non-eager / 757 tok | 8 | 39.0% | 39% | 287712 | 287712 |

This is not a fine-grained profiler, but it supports the main reading: non-eager runs are visibly driving the device harder and achieving higher wall throughput.

## What This Teaches

The main learning point is that "vLLM performance" is not one number. It changes depending on which phase dominates:

1. Startup: model load, compile/cache state, graph capture.
2. Prefill: prompt length and batching dominate TTFT-like latency.
3. Decode: per-token loop benefits from graph replay and stable buffers.
4. KV/cache capacity: decides how many live requests can remain resident.
5. Scheduler behavior: concurrency improves throughput until latency and batching overhead become the limiting factor.

For an inference optimization internship story, this is a strong next pitch:

> I used a MetaX C500 vLLM image to compare eager and non-eager serving for LFM2.5. The non-eager path entered vLLM compile and CUDAGraph capture, paying slightly higher startup in a warm-cache run but roughly doubling steady-state completion-token throughput. Then I increased prompt length to separate prefill pressure from decode throughput: longer prompts raised request latency and reduced completion-token throughput at higher concurrency, even though total token throughput increased because prefill tokens dominated the count.

## Next Experiments

- Add streaming benchmark to split TTFT and TPOT instead of using end-to-end latency.
- Repeat each run 3 times and report median/p90 across runs.
- Run the same harness on Qwen3-0.6B as a pure-attention baseline.
- Add `max_model_len` and `gpu_memory_utilization` sweeps to API server mode, not only offline lifecycle mode.
- Use vendor profiler / trace tools if available; `mx-smi` is enough for evidence, not root-cause analysis.
