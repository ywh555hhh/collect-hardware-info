# MetaX C500 vLLM deeper exploration

采集时间：2026-08-12

这轮继续探索三个问题：

1. LFM2 hybrid 模型和 Qwen3 pure attention 模型的 KV cache 行为有什么不同？
2. 如果保持 API server 常驻，稳态请求延迟和吞吐大概长什么样？
3. 不加 `--enforce-eager` 时，MACA/vLLM 是否能进入 torch.compile + CUDAGraph 路径？

结论先讲：这轮数据已经很适合作为推理优化面试材料，因为它不只是“能跑模型”，而是能解释 vLLM 如何根据模型架构、显存预算和上下文长度做 serving capacity planning。

## Artifacts

| Experiment | Raw data |
| --- | --- |
| LFM2 KV sweep | `raw/metax-c500-vllm/kv_sweep_lfm25/summary.json` |
| Qwen3-0.6B KV sweep | `raw/metax-c500-vllm/kv_sweep_qwen3_0p6b/summary.json` |
| LFM2 API bench | `raw/metax-c500-vllm/api_bench_lfm25/lfm25_api_bench.json` |
| LFM2 non-eager probe | `raw/metax-c500-vllm/non_eager_probe/lfm25_non_eager.json` |
| Harness | `scripts/vllm_lifecycle_probe.py`, `scripts/vllm_kv_sweep.py`, `scripts/vllm_api_bench.py` |

## LFM2 vs Qwen3 KV Cache

Same sweep shape:

- `max_model_len`: `1024, 2048, 4096`
- `gpu_memory_utilization`: `0.50, 0.70`
- `batch_size`: `1`
- `max_tokens`: `16`
- `enforce_eager`: enabled

### LFM2-1.2B-Instruct

| max_model_len | gpu_mem_util | KV avail GiB | KV tokens | max concurrency |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.50 | 4.42 | 128,608 | 365.38x |
| 1024 | 0.70 | 7.46 | 217,264 | 617.24x |
| 2048 | 0.50 | 4.42 | 128,608 | 185.50x |
| 2048 | 0.70 | 7.46 | 217,264 | 313.37x |
| 4096 | 0.50 | 4.42 | 128,608 | 93.47x |
| 4096 | 0.70 | 7.46 | 217,264 | 157.90x |

Runtime-specific signals:

```text
Setting attention block size to 16 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 300.00% to ensure that mamba page size and attention page size are exactly equal.
Add 2 padding layers, may waste at most 20.00% KV cache memory.
```

### Qwen3-0.6B

| max_model_len | gpu_mem_util | KV avail GiB | KV tokens | max concurrency |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.50 | 4.72 | 44,224 | 43.19x |
| 1024 | 0.70 | 7.77 | 72,720 | 71.02x |
| 2048 | 0.50 | 4.72 | 44,224 | 21.59x |
| 2048 | 0.70 | 7.77 | 72,720 | 35.51x |
| 4096 | 0.50 | 4.72 | 44,224 | 10.80x |
| 4096 | 0.70 | 7.77 | 72,720 | 17.75x |

Qwen3 does not emit the Mamba/page-alignment or padding-layer logs. It follows the ordinary attention path:

```text
Using FLASH_ATTN attention backend
```

## Interpretation: Why the Difference Is Interesting

At first glance, Qwen3-0.6B is smaller than LFM2-1.2B, so one might expect Qwen3 to have higher concurrency. The vLLM logs show the opposite: LFM2 reports far more cache tokens and much higher max concurrency.

The reason is that serving capacity is not just a function of parameter count. It depends on the per-token runtime state layout:

- Qwen3 is a standard attention model, so serving long context means reserving K/V states across attention layers.
- LFM2 is hybrid. vLLM has to align attention pages with Mamba/state-space pages, but the reported token capacity is still much larger for this specific model/config.
- Therefore, KV/cache planning is model-architecture dependent. A smaller weight file does not automatically imply lower cache pressure, and a larger model does not automatically imply lower request concurrency.

This is an excellent interview talking point: inference optimization starts by asking what state must live per request, not just how many parameters the model has.

## Steady-State API Bench

Model: `LFM2.5-1.2B-Instruct`

Settings:

- `max_model_len=2048`
- `gpu_memory_utilization=0.70`
- `max_tokens=32`
- `enforce_eager=True`
- one API server kept alive across all requests

Startup:

| Metric | Value |
| --- | ---: |
| API server startup | 60.693s |

Request results:

| concurrency | requests | ok | wall_s | completion tok/s wall | latency p50 ms | latency mean ms | latency max ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 4 | 1.841 | 69.525 | 292.167 | 459.687 | 976.565 |
| 2 | 4 | 4 | 1.141 | 112.173 | 837.451 | 568.087 | 845.624 |
| 4 | 4 | 4 | 0.327 | 391.292 | 325.843 | 322.629 | 326.608 |
| 8 | 4 | 4 | 0.325 | 394.139 | 323.077 | 320.440 | 323.215 |

This is still a tiny smoke bench, but it demonstrates an important lifecycle distinction: repeated model startup is not serving performance. Once the API server is warm, wall throughput is much higher than the one-shot offline lifecycle numbers suggest.

The odd-looking latency pattern at concurrency 1/2 is likely warmup / first-request effects and tiny sample size. The next benchmark should use more requests per concurrency level and discard warmup requests.

## Eager vs Non-Eager Probe

The earlier experiments used `--enforce-eager`, which disables torch.compile and CUDAGraph. This probe removed `--enforce-eager` for LFM2.

Result: non-eager works on this C500/vllm_metax stack.

Key logs:

```text
enforce_eager=False
compilation_config mode: VLLM_COMPILE
cudagraph_mode: FULL_AND_PIECEWISE
torch.compile takes 23.83 s in total
Graph capturing finished in 4 secs, took 0.15 GiB
```

Offline result:

| Mode | load wall s | generate wall s | smoke tok/s |
| --- | ---: | ---: | ---: |
| eager | ~48.948 | ~0.833 | ~19.210 |
| non-eager | 75.477 | 0.639 | 25.040 |

API server result:

| Mode | startup s | one request ms |
| --- | ---: | ---: |
| eager | 60.693 | not directly same harness |
| non-eager | 82.884 | 441.6 |

Interpretation:

- Non-eager has higher startup cost because it compiles and captures graphs.
- It does not fail on this stack, which is important: MACA/vllm_metax is not limited to eager-only smoke mode.
- The small generation sample suggests possible steady-state benefit, but the sample is too small to claim performance improvement.
- A serious next step is a warm server benchmark comparing eager vs non-eager with many requests and fixed prompt/output lengths.

## What This Teaches

A good inference optimization investigation separates five layers:

1. Model architecture: pure attention vs hybrid/state-space changes per-request state.
2. Memory budget: `gpu_memory_utilization` determines how much memory vLLM can reserve for runtime cache.
3. Context length: `max_model_len` determines how expensive one resident request can become.
4. Execution mode: eager is a compatibility path; non-eager introduces compile + graph capture startup cost but may improve steady-state decode.
5. Serving mode: one-shot `LLM()` smoke is not the same as a warm API server with concurrent requests.

The best short pitch from this round:

> I compared LFM2 and Qwen3 on C500/vllm_metax. Changing `gpu_memory_utilization` changed total KV/cache capacity, while changing `max_model_len` reduced max concurrency. LFM2 additionally triggered Mamba/attention page-size alignment and padding-layer logs, showing that hybrid models have different cache planning constraints from pure attention models. I also verified that non-eager mode enters torch.compile and CUDAGraph capture on MACA, trading slower startup for a more optimized serving path.

## Next Moves

- Run warm API benchmark with more requests, discard first request as warmup.
- Compare eager vs non-eager API server under identical concurrency levels.
- Add prompt length sweep: 128 / 512 / 2048 tokens to separate prefill and decode behavior.
- Sample `mx-smi --show-ap-usage` and `--show-hbm-bandwidth` during active serving.
- Add SmolLM3-3B as a standard modern decoder-only model between Qwen3-0.6B and Qwen3-4B.
