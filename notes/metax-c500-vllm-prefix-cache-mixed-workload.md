# MetaX C500 vLLM prefix cache / mixed prompt workload

采集时间：2026-08-12

这轮实验专门验证一个真实 serving 场景：

> 多个请求共享一段很长的 system/context prefix，只在最后的 user query 有差异时，vLLM prefix cache 是否能降低 TTFT，并改善 mixed workload 的吞吐表现？

这比固定 prompt benchmark 更接近生产服务，例如：

- RAG 系统共享同一段检索上下文模板
- Agent 服务共享同一段 tool/system prompt
- 多轮对话共享历史前缀
- 批量评测共享题干或 rubric

## Artifacts

| Experiment | Raw data |
| --- | --- |
| Qwen3-0.6B prefix cache, concurrency 1 | `raw/metax-c500-vllm/prefix_cache_qwen3/qwen3_0p6b_prefix_cache_c1.json` |
| Qwen3-0.6B prefix cache, concurrency 8 | `raw/metax-c500-vllm/prefix_cache_qwen3/qwen3_0p6b_prefix_cache_c8_v2.json` |
| First c8 attempt with unfair unique prompt length | `raw/metax-c500-vllm/prefix_cache_qwen3/qwen3_0p6b_prefix_cache_mixed.json` |
| Harness | `scripts/vllm_prefix_cache_probe.py` |

Important caveats:

- This is API-level timing, not a kernel profiler.
- TTFT is measured as first streamed content chunk latency.
- TPOT is approximated from full streamed latency and `usage.completion_tokens`.
- `mx-smi` samples are too coarse for the main conclusion; the useful evidence is TTFT/TPOT plus vLLM config/logs.
- The first c8 attempt is kept as raw data, but main conclusions below use the corrected c1 and c8_v2 runs.

## Environment

| Item | Value |
| --- | --- |
| GPU | MetaX C500, current sGPU slice |
| Visible memory | ~15.6GB |
| Model | Qwen3-0.6B |
| vLLM | 0.17.0 |
| vllm_metax | 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8 |
| Execution mode | non-eager, `torch.compile` + CUDAGraph |
| Serving | OpenAI-compatible `/v1/chat/completions`, `stream=true` |
| `max_model_len` | 2048 |
| `gpu_memory_utilization` | 0.70 |
| `max_tokens` | 32 |

The server config confirms prefix caching is enabled:

```text
enable_prefix_caching=True
block_size="16"
prefix_caching_hash_algo="sha256"
```

The concurrency-1 run also emitted a direct log signal:

```text
Prefix cache hit rate: 66.7%
```

## Workload Design

The harness runs four stages inside one server process:

| Stage | Purpose |
| --- | --- |
| `unique_prefix_control` | Each request changes the first prefix block, preventing long prefix reuse |
| `shared_prefix_cold` | Requests share the same long prefix before explicit warmup |
| `shared_prefix_warmup` | Sequential shared-prefix requests to populate cache |
| `shared_prefix_cached` | Shared-prefix mixed requests after warmup |

The corrected control keeps prompt lengths close:

- concurrency 1: unique ~988 prompt tokens vs shared ~1017 prompt tokens
- concurrency 8: unique ~988 prompt tokens vs shared ~1017 prompt tokens

The earlier c8 attempt had unique ~1276 tokens vs shared ~1017, so it is not used for the main comparison.

## Concurrency 1 Results

| stage | ok | avg prompt tokens | TTFT p50 ms | TTFT p90 ms | TPOT p50 ms | completion tok/s wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unique_prefix_control | 8 | 987.75 | 25.358 | 28.258 | 4.369 | 189.474 |
| shared_prefix_cold | 8 | 1016.75 | 16.579 | 22.617 | 4.375 | 206.260 |
| shared_prefix_warmup | 4 | 1016.75 | 16.681 | 17.228 | 4.401 | 196.263 |
| shared_prefix_cached | 8 | 1016.75 | 16.310 | 16.757 | 4.394 | 208.963 |

Per-request TTFT tells the story:

- `unique_prefix_control`: first request was ~80ms, later requests mostly ~25ms.
- `shared_prefix_cold`: first two shared requests were ~29ms / ~23ms, then later requests fell to ~16ms.
- `shared_prefix_cached`: all measured requests were stable around ~15-17ms.

Interpretation:

- Prefix cache primarily improves TTFT / prefill cost, not TPOT. TPOT stayed around ~4.37-4.40ms/token.
- Once the shared prefix is resident, first-token latency stabilizes around ~16ms.
- The 66.7% hit-rate log confirms vLLM observed prefix-cache reuse in this run.

## Concurrency 8 Results

| stage | ok | avg prompt tokens | TTFT p50 ms | TTFT p90 ms | TPOT p50 ms | completion tok/s wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unique_prefix_control | 16 | 988.13 | 112.123 | 178.178 | 6.938 | 750.241 |
| shared_prefix_cold | 16 | 1016.75 | 54.489 | 60.087 | 5.336 | 1142.529 |
| shared_prefix_warmup | 8 | 1016.75 | 16.911 | 17.091 | 4.371 | 209.374 |
| shared_prefix_cached | 16 | 1016.75 | 48.665 | 61.625 | 5.519 | 1152.605 |

Interpretation:

- Under concurrency 8, shared-prefix requests cut TTFT p50 by more than half versus unique-prefix control: ~49-54ms vs ~112ms.
- Completion throughput improves from ~750 tok/s to ~1150 tok/s.
- The cached and cold shared-prefix stages are close under concurrency 8, which suggests vLLM can benefit quickly once identical-prefix requests arrive; explicit warmup adds only modest improvement in this small test.
- TPOT also improves versus unique control, but the dominant effect is reduced prefill/TTFT.

## Why This Matters

Prefix cache is not just a micro-optimization. It changes the economics of real LLM applications:

1. Long shared prompt prefixes become cheaper after the first request.
2. TTFT is the most visible user-facing gain.
3. Decode TPOT does not change much, because the generated suffix still has to run token by token.
4. Scheduler behavior matters: under concurrency, prefix-cache gains mix with batching and queueing.
5. Prompt layout becomes a systems decision: put reusable context before request-specific text.

The systems mental model:

```text
without prefix cache:
  tokenize -> prefill full prompt -> decode

with shared prefix cache:
  tokenize -> reuse cached prefix KV blocks -> prefill suffix -> decode
```

The important constraint is prefix order: vLLM can reuse identical prefix blocks from the beginning of the prompt. If a request-specific marker appears too early, long-prefix reuse disappears.

## Interview Pitch

> I built a mixed workload benchmark on C500/vLLM to test prefix caching. The workload compares unique long prefixes against shared long prefixes with different user suffixes. With Qwen3-0.6B, prefix caching was enabled with 16-token blocks and SHA256 hashing; the server log reported a 66.7% prefix cache hit rate in the single-concurrency run. In the corrected concurrency-8 run, shared-prefix requests reduced TTFT p50 from ~112ms to ~49-54ms and improved completion throughput from ~750 tok/s to ~1150 tok/s. TPOT moved less, which matches the model that prefix cache primarily removes repeated prefill work rather than speeding up decode.

## Next Experiments

- Sweep shared prefix length: 128 / 512 / 1024 / 1536 prompt tokens.
- Put the unique marker early vs late to quantify prompt-layout impact.
- Try mixed arrival processes instead of fixed concurrency groups.
- Repeat on LFM2.5 and document why prefix caching is disabled or unsupported there if it remains off.
- Parse or expose richer vLLM prefix-cache metrics if the current Prometheus output does not include internal token-hit counters.
