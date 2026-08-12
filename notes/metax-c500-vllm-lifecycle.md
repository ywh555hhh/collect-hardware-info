# MetaX C500 vLLM lifecycle notes

采集时间：2026-08-12

这份 note 记录 C500 的 vLLM 镜像在“模型生命周期”上的表现：环境识别、模型获取、vLLM offline `LLM()`、OpenAI-compatible API server、以及失败/异常原因。

## 镜像栈

| 项目 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.3 LTS |
| GPU | MetaX C500, sGPU 25% compute / 15.625GB visible VRAM |
| MACA | 3.5.3.20 |
| driver | KMD 3.8.30 |
| Python | 3.10.10 |
| torch | 2.8.0+metax3.5.3.9 |
| vLLM | 0.17.0 |
| vllm_metax | 0.17.0+gd10261.d20260409.maca3.5.3.20.torch2.8 |
| transformers | 4.57.6 |
| flash_attn | 2.6.3+metax3.5.3.9torch2.8 |
| triton | 3.0.0+metax3.5.3.9 |

关键镜像行为：

- `vllm_metax` 能作为 `vllm.platform_plugins` 自动加载。
- 插件会设置 `VLLM_USE_FLASHINFER_SAMPLER=False`，原因是 MACA 上不支持 FlashInfer sampler。
- 插件会把 `VLLM_ENGINE_READY_TIMEOUT_S` 设置为 `3600`，说明模型加载/初始化可能显著慢于常规 CUDA 镜像。
- 插件会覆盖 AWQ/GPTQ/compressed-tensors 等 quantization config，并注册 DeepSeek/Kimi/Step 等模型实现。
- 插件会使用 MACA 版 flash attention，日志显示 “Using Maca version of flash attention, which only supports version 2”。

## 已测模型

| 模型 | 路径 | 结果 | load_s | generate_s | new tokens | aggregate tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | `/mnt/moark-models/Qwen3-0.6B` | OK | 78.394 | 5.273 | 192 | 36.411 |
| Qwen3-4B | `/mnt/moark-models/Qwen3-4B` | OK | 79.851 | 1.256 | 192 | 152.812 |
| LFM2.5-1.2B-Instruct | `/data/models/LFM2.5-1.2B-Instruct` | OK | 75.920 | 2.953 | 49 | 16.596 |

注意：上面的吞吐只是 smoke 数据，不是严肃 benchmark。Qwen3-4B 在这个短样本里比 0.6B 高，不应解读为 4B 更快，主要受 prompt、输出长度、调度状态、首次运行和采样路径影响。

## LFM2.5 观察

LFM2.5-1.2B-Instruct 是这次更有价值的新模型样本，因为它不是普通 Llama/Qwen 形态。vLLM 解析到：

```json
{
  "model_type": "lfm2",
  "architectures": ["Lfm2ForCausalLM"],
  "hidden_size": 2048,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 128000,
  "rope_theta": 1000000.0
}
```

vLLM 生命周期中出现的关键日志：

- `Resolved architecture: Lfm2ForCausalLM`
- `Setting attention block size to 16 tokens to ensure that attention page size is >= mamba page size.`
- `Padding mamba page size by 300.00% to ensure that mamba page size and attention page size are exactly equal.`
- `enable_prefix_caching=False`
- `Add 2 padding layers, may waste at most 20.00% KV cache memory`
- `GPU KV cache size: 217,264 tokens`
- `Maximum concurrency for 2,048 tokens per request: 313.37x`

这说明这个模型在 vLLM 里触发了 hybrid/state-space 相关的 page 对齐逻辑，而不是普通纯 attention 模型路径。它能在 C500 + vllm_metax 上跑通，说明这个镜像对新模型架构的支持面比“只跑 Qwen/Llama”更宽。

## API server 闭环

使用 LFM2.5-1.2B-Instruct 启动 OpenAI-compatible API server：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/LFM2.5-1.2B-Instruct \
  --served-model-name LFM2.5-1.2B-Instruct \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000
```

结果：

| 项目 | 值 |
| --- | --- |
| `/v1/models` | OK |
| `/v1/chat/completions` | OK |
| startup_s | 62 |
| request_ms | 1478 |
| prompt_tokens | 27 |
| completion_tokens | 36 |
| total_tokens | 63 |

返回内容：

> 在C500上测试vLLM API服务的意义在于验证其在高并发、大规模推理场景下的稳定性和性能表现。

## 生命周期问题和解释

### 1. `spawn` multiprocessing 限制

第一版脚本在顶层直接创建 `LLM()`，触发了 Python multiprocessing 的经典错误：

```text
An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

原因不是 C500 不支持，也不是 vLLM 不支持，而是 `vllm_metax` 会强制使用 `spawn` multiprocessing start method。脚本必须使用：

```python
if __name__ == "__main__":
    main()
```

### 2. 下载是生命周期的一部分

LFM2.5-1.2B-Instruct 从 HF mirror 下载约 2.34GB 权重，实际过程比较慢，中间 SSH 会话断开，但远端 `hf download` 进程继续运行，最终完成。这说明国产云上的实验需要把模型获取、缓存目录、锁文件和断点续传也纳入记录。

### 3. eager 模式降低了性能意义

本次命令使用 `--enforce-eager`，日志明确显示：

```text
disabling torch.compile and CUDAGraphs
Cudagraph is disabled under eager mode
```

所以这次数据适合判断“能不能完整跑通”，不适合判断 C500/vLLM 的峰值性能。下一步 benchmark 应该分成 eager smoke 和 graph/compile performance 两组。

## 原始数据

- `raw/metax-c500-vllm/c500_vllm_lifecycle_probe.json`
- `raw/metax-c500-vllm/c500_vllm_lifecycle_probe.log`
- `raw/metax-c500-vllm/c500_new_model_download_probe.log`
- `raw/metax-c500-vllm/c500_lfm25_lifecycle.json`
- `raw/metax-c500-vllm/c500_lfm25_lifecycle.log`
- `raw/metax-c500-vllm/c500_lfm25_api_server_summary.txt`
- `raw/metax-c500-vllm/c500_lfm25_api_server.log`
- `raw/metax-c500-vllm/c500_lfm25_api_response.json`

## 后续建议

- 补 SmolLM3-3B 下载和 vLLM 生命周期，作为标准 decoder-only 小模型对照组。
- 对 LFM2.5 分别测试 `--enforce-eager` 和非 eager 模式，观察 CUDAGraph/compile 在 MACA 上是否可用。
- 做固定输入长度/固定输出长度/固定 batch 的吞吐测试，不再用短 prompt smoke 数据解释性能。
- 记录 `mx-smi --show-hbm-bandwidth` 和 `--show-ap-usage` 在推理期间的变化。
