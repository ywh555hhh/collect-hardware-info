# MetaX C500 vLLM streaming TTFT / TPOT benchmark

采集时间：2026-08-12

这轮实验把 API benchmark 从 non-streaming end-to-end latency 推进到 streaming metrics：

- TTFT-like metric: first content chunk latency, recorded as `first_content_ms`
- TPOT-like metric: `(full latency - first_content_ms) / (completion_tokens - 1)`, recorded as `tpot_ms`
- Full latency: entire streamed response completion time
- Throughput: wall-clock completion tokens/s across each concurrency group

这比只看完整响应延迟更接近推理优化面试里真正关心的拆法：prefill 影响 first token，decode loop 影响 per-token latency。

## Artifacts

| Experiment | Raw data |
| --- | --- |
| LFM2.5 non-eager streaming, prompt ~=237 tokens | `raw/metax-c500-vllm/api_bench_lfm25_stream/lfm25_noneager_stream_prompt128.json` |
| LFM2.5 non-eager streaming, prompt ~=757 tokens | `raw/metax-c500-vllm/api_bench_lfm25_stream/lfm25_noneager_stream_prompt512.json` |
| Qwen3-0.6B non-eager streaming, prompt ~=176 tokens | `raw/metax-c500-vllm/api_bench_qwen3_stream/qwen3_0p6b_noneager_stream_prompt128.json` |
| Harness | `scripts/vllm_api_bench.py` |

Notes:

- `--prompt-token-len` is approximate. The actual token counts below come from vLLM `usage.prompt_tokens`.
- This is still a pragmatic benchmark, not a controlled profiler. It uses OpenAI-compatible streaming and Python client timing.
- The first content chunk is a good TTFT proxy, but it includes HTTP/client overhead.
- TPOT is computed from API usage token counts and full streamed latency, not from kernel traces.

## Environment

| Item | Value |
| --- | --- |
| GPU | MetaX C500, current sGPU slice |
| Visible memory | ~15.6GB |
| MACA | 3.5.3.20 |
| vLLM | 0.17.0 |
| vllm_metax | 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8 |
| PyTorch | 2.8.0+metax3.5.3.9 |
| Serving | OpenAI-compatible `/v1/chat/completions`, `stream=true` |
| Execution mode | non-eager, `torch.compile` + CUDAGraph |
| `max_model_len` | 2048 |
| `gpu_memory_utilization` | 0.70 |
| `max_tokens` | 64 |

Both LFM2.5 and Qwen3 used the MACA flash attention backend:

```text
Using FLASH_ATTN attention backend
Using Maca version of flash attention, which only supports version 2.
```

Both runs entered the non-eager compiled path:

```text
compilation_config mode: VLLM_COMPILE
cudagraph_mode: FULL_AND_PIECEWISE
Graph capturing finished
```

## LFM2.5 Streaming: Short Prompt

Model: `LFM2.5-1.2B-Instruct`

Actual prompt tokens: ~237

| concurrency | ok | TTFT p50 ms | TTFT p90 ms | TPOT p50 ms | full latency p50 ms | completion tok/s wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 15.686 | 15.968 | 4.031 | 225.437 | 234.001 |
| 2 | 16 | 18.496 | 18.705 | 4.571 | 269.926 | 409.919 |
| 4 | 16 | 27.931 | 36.445 | 4.604 | 267.312 | 785.719 |
| 8 | 16 | 45.368 | 50.690 | 4.870 | 299.013 | 1403.259 |
| 16 | 16 | 63.283 | 85.988 | 5.719 | 359.520 | 2257.624 |

Interpretation:

- Throughput scales strongly with concurrency: ~234 -> ~2258 completion tok/s from concurrency 1 to 16.
- TTFT rises with concurrency: ~16ms -> ~63ms p50. That is the visible cost of batching/scheduling more resident work.
- TPOT rises more gently: ~4.0ms/token -> ~5.7ms/token. Decode remains efficient, but higher concurrency adds scheduling and memory pressure.

## LFM2.5 Streaming: Longer Prompt

Model: `LFM2.5-1.2B-Instruct`

Actual prompt tokens: ~757

| concurrency | ok | TTFT p50 ms | TTFT p90 ms | TPOT p50 ms | full latency p50 ms | completion tok/s wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 21.015 | 21.758 | 3.995 | 272.644 | 234.386 |
| 2 | 12 | 32.537 | 33.740 | 4.549 | 309.428 | 412.194 |
| 4 | 12 | 50.484 | 59.525 | 4.696 | 346.281 | 726.274 |
| 8 | 12 | 66.606 | 120.244 | 5.201 | 420.954 | 991.345 |

Interpretation:

- Longer prompt mainly pushes up TTFT/full latency, because prefill has more tokens to process before first output.
- At concurrency 8, TTFT p90 jumps to ~120ms; this is the clearest sign of prefill and scheduler pressure.
- TPOT is close to the short-prompt run at the same concurrency. This is the expected shape: prompt length changes prefill more than decode.

## Qwen3-0.6B Streaming Baseline

Model: `Qwen3-0.6B`

Actual prompt tokens: ~176

| concurrency | ok | TTFT p50 ms | TTFT p90 ms | TPOT p50 ms | full latency p50 ms | completion tok/s wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 13.132 | 13.633 | 4.304 | 284.275 | 224.498 |
| 2 | 16 | 19.408 | 20.622 | 4.639 | 309.888 | 411.874 |
| 4 | 16 | 20.778 | 30.458 | 4.679 | 316.570 | 804.207 |
| 8 | 16 | 31.042 | 32.283 | 4.977 | 344.633 | 1478.776 |
| 16 | 16 | 42.373 | 49.050 | 5.471 | 386.044 | 2602.080 |

Startup and cache capacity are where Qwen3 looks very different from LFM2.5:

| model | startup_s | torch.compile s | KV cache tokens | max concurrency at 2048 tokens |
| --- | ---: | ---: | ---: | ---: |
| LFM2.5-1.2B | 70.430 | 4.98 | 217,952 | 314.35x |
| Qwen3-0.6B | 122.316 | 49.58 | 72,832 | 35.56x |

Interpretation:

- Qwen3 has lower TTFT at high concurrency and slightly higher completion throughput at concurrency 16.
- But Qwen3 reports much lower KV cache token capacity and max concurrency for 2048-token requests.
- This is exactly the useful systems lesson: smaller parameter count does not automatically mean better serving capacity. Runtime state layout, attention architecture, compile behavior, and cache planning all matter.

## Hardware Sampling

The harness also sampled `mx-smi --show-usage`, `--show-ap-usage`, and `--show-hbm-bandwidth` during measured groups.

Representative samples:

| run | concurrency | avg GPU util | max GPU util | avg HBM MB/s | max HBM MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFM2.5 / 237 tok | 1 | 38.8% | 41% | 579547 | 589990 |
| LFM2.5 / 237 tok | 8 | 37.0% | 37% | 485632 | 485632 |
| LFM2.5 / 757 tok | 1 | 40.5% | 43% | 582004 | 608853 |
| LFM2.5 / 757 tok | 8 | 41.0% | 41% | 284843 | 284843 |
| Qwen3 / 176 tok | 1 | 31.6% | 32% | 289162 | 293987 |
| Qwen3 / 176 tok | 8 | 29.0% | 29% | 264073 | 264073 |

These counters are coarse, but they support the qualitative observation: LFM2.5 drives higher device utilization and HBM bandwidth in these serving shapes than Qwen3-0.6B.

## What This Teaches

The useful mental model from this round:

1. TTFT is mostly about request admission, tokenization/chat-template overhead, prefill, and scheduler queueing.
2. TPOT is mostly about the decode loop: model forward, KV/state reads, sampling, and graph replay overhead.
3. Higher concurrency improves throughput but raises TTFT and TPOT.
4. Longer prompts mostly hurt TTFT/full latency, not per-token decode cost.
5. Model architecture changes serving capacity: LFM2.5 has higher reported KV/cache capacity, while Qwen3 has slightly faster high-concurrency streaming throughput in this narrow run.

The interview-ready pitch:

> I extended a C500/vLLM benchmark from full-response latency to streaming TTFT and TPOT. On LFM2.5, increasing concurrency improved completion-token throughput from ~234 to ~2258 tok/s, while TTFT rose from ~16ms to ~63ms and TPOT from ~4.0ms/token to ~5.7ms/token. Increasing prompt length mostly raised TTFT and full latency, which separates prefill pressure from decode cost. I then compared Qwen3-0.6B: it reached ~2602 tok/s at concurrency 16, but had much lower reported KV cache capacity, showing why parameter count alone is not a serving-capacity metric.

## Next Experiments

- Repeat each streaming run 3 times and report median across runs.
- Add Qwen3-4B if memory permits, to separate "small model" effects from architecture effects.
- Sweep `max_model_len` under streaming API mode to see whether TTFT/TPOT change before cache pressure becomes visible.
- Add request arrival patterns instead of fixed concurrency groups: Poisson arrivals, burst arrivals, and mixed prompt lengths.
- If vendor profiling tools are available, trace prefill/decode kernel time instead of relying only on API-level timing.
