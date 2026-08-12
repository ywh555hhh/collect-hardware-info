# MetaX C500 vLLM lifecycle and KV cache pilot

采集时间：2026-08-12

这份 note 把两个方向合在一起：

1. vLLM 生命周期拆解：从模型配置解析到权重加载、backend 选择、KV cache 分配、warmup、generate/API request。
2. KV cache / page size / concurrency 分析：改变 `max_model_len` 和 `gpu_memory_utilization`，观察 KV pool 和最大并发如何变化。

本轮是 pilot，不是性能 benchmark。所有 case 都使用 `--enforce-eager`，因此 torch.compile / CUDAGraph 被关闭，结果适合解释生命周期和显存调度，不适合宣称峰值吞吐。

## Tested Model

模型：`LFM2.5-1.2B-Instruct`

路径：`/data/models/LFM2.5-1.2B-Instruct`

选择原因：

- 体量小，权重约 2.34GB，适合 15.6GB 可见显存的 C500 sGPU slice。
- 不是普通 Llama/Qwen decoder-only 路线，vLLM 日志会暴露 hybrid / mamba page-size 相关逻辑。
- 已验证 offline `LLM()` 和 OpenAI-compatible API server 均可跑通。

模型配置摘要：

| Field | Value |
| --- | --- |
| `model_type` | `lfm2` |
| `architectures` | `Lfm2ForCausalLM` |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 16 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `max_position_embeddings` | 128000 |
| `rope_theta` | 1000000.0 |

## Lifecycle Signals

一次 vLLM 启动不是简单的 “load model”，实际能拆成这些阶段：

| Stage | Evidence from logs | Meaning |
| --- | --- | --- |
| Plugin registration | `Platform plugin metax is activated` | vLLM 自动加载 `vllm_metax` 后端 |
| Runtime policy | `VLLM_USE_FLASHINFER_SAMPLER=False` | MACA 不走 FlashInfer sampler |
| Model config parse | `Resolved architecture: Lfm2ForCausalLM` | vLLM 能识别 LFM2 架构 |
| Dtype normalization | `Casting torch.bfloat16 to torch.float16` | 本轮强制用 fp16 |
| Scheduler config | `Chunked prefill is enabled` | prefill 会被 chunk 化调度 |
| Hybrid page alignment | `attention page size >= mamba page size` | LFM2 触发 mamba/attention page 对齐 |
| Backend selection | `Using FLASH_ATTN attention backend` | attention 走 MACA flash attention |
| Weight load | `Loading weights took ~22-24s` | safetensors 权重加载阶段 |
| Model memory | `Model loading took 2.2 GiB memory` | 权重/模型常驻显存 |
| KV allocation | `Available KV cache memory` + `GPU KV cache size` | 剩余显存被分配给 KV/state cache |
| Concurrency estimate | `Maximum concurrency for N tokens/request` | 给定上下文长度下的理论并发上限 |
| Warmup | `init engine ... took ~1.9-2.1s` | profiling、KV cache 创建、模型 warmup |
| Request | `generate` / `/v1/chat/completions` | 真正请求路径 |

## KV Cache Sweep

Sweep 参数：

- `max_model_len`: `1024, 2048, 4096`
- `gpu_memory_utilization`: `0.50, 0.70`
- `batch_size`: `1`
- `max_tokens`: `16`
- `enforce_eager`: enabled

Raw output:

- `raw/metax-c500-vllm/kv_sweep_lfm25/summary.json`
- `raw/metax-c500-vllm/kv_sweep_lfm25/*.log`
- `raw/metax-c500-vllm/kv_sweep_lfm25/*.json`

Summary:

| max_model_len | gpu_mem_util | KV avail GiB | KV tokens | context tokens | max concurrency | load wall s | smoke tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.50 | 4.42 | 128,608 | 1024 | 365.38x | 47.166 | 12.358 |
| 1024 | 0.70 | 7.46 | 217,264 | 1024 | 617.24x | 48.217 | 19.163 |
| 2048 | 0.50 | 4.42 | 128,608 | 2048 | 185.50x | 47.942 | 19.474 |
| 2048 | 0.70 | 7.46 | 217,264 | 2048 | 313.37x | 48.948 | 19.210 |
| 4096 | 0.50 | 4.42 | 128,608 | 4096 | 93.47x | 48.950 | 19.235 |
| 4096 | 0.70 | 7.46 | 217,264 | 4096 | 157.90x | 47.078 | 19.557 |

## Interpretation

### 1. `gpu_memory_utilization` controls the KV/state pool size

At `gpu_memory_utilization=0.50`, vLLM reports:

- available KV cache memory: `4.42 GiB`
- GPU KV cache size: `128,608 tokens`

At `gpu_memory_utilization=0.70`, vLLM reports:

- available KV cache memory: `7.46 GiB`
- GPU KV cache size: `217,264 tokens`

The model memory stays near `2.2 GiB`, so the extra budget mostly becomes KV/state cache. This is the practical meaning of `gpu_memory_utilization`: it is not "make the model faster" directly; it changes how much memory vLLM reserves for runtime serving capacity.

### 2. `max_model_len` changes per-request budget, not total KV tokens

For fixed `gpu_memory_utilization`, total KV tokens are stable:

- `0.50`: always `128,608 tokens`
- `0.70`: always `217,264 tokens`

But reported max concurrency drops as `max_model_len` grows:

- `1024 -> 2048 -> 4096`
- `365.38x -> 185.50x -> 93.47x` at `gpu_memory_utilization=0.50`
- `617.24x -> 313.37x -> 157.90x` at `gpu_memory_utilization=0.70`

This is the central vLLM trade-off: a longer context window reduces the number of requests that can be resident at once. You are spending the same cache pool with larger per-request reservations.

### 3. LFM2 has extra hybrid-cache constraints

Every case emits:

```text
Setting attention block size to 16 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 300.00% to ensure that mamba page size and attention page size are exactly equal.
Add 2 padding layers, may waste at most 20.00% KV cache memory.
```

For a pure attention model, KV cache analysis is mostly about K/V pages. For LFM2, vLLM also has to align mamba/state-space page size with attention page size. This is why LFM2 is a better interview example than only Qwen3: it shows that modern inference engines are not just "paged attention for every model"; hybrid architectures introduce their own cache layout constraints.

### 4. Load time dominates this pilot

Each case takes roughly `47-49s` wall-clock load time, while generation is under `1.3s` for a tiny 16-token smoke output. That means this pilot is not measuring steady-state serving performance. It is measuring startup lifecycle and cache planning.

The next benchmark should keep the server alive and issue many requests, otherwise repeated model load time dominates the experiment.

## Interview Angle

This experiment is useful to explain the difference between "running a model" and "understanding an inference runtime":

- Application view: send request, get tokens.
- Inference infra view: choose backend, allocate KV/state cache, estimate concurrency, schedule prefill/decode, manage page sizes, decide graph/eager mode.

The strongest talking point is:

> I changed `max_model_len` and `gpu_memory_utilization` on a C500 vLLM-MetaX stack, then showed that total cache capacity follows memory budget while max concurrency falls roughly inversely with request context length. On LFM2, vLLM also has to align Mamba and attention page sizes, so the cache story is richer than plain paged attention.

## Next Experiments

- Run the same sweep on Qwen3-0.6B as a pure attention baseline.
- Keep one API server alive and issue repeated requests to separate startup cost from steady-state TTFT/TPOT.
- Add `mx-smi --show-ap-usage` and `mx-smi --show-hbm-bandwidth` sampling while requests are active.
- Compare eager vs non-eager mode to see whether MACA supports graph/compile paths cleanly.
- Increase batch/request concurrency and measure TTFT/TPOT rather than only aggregate smoke tokens/s.
