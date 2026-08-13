# 简历项目稿：FastDeploy / C500 / 4090 / MR-V100 推理框架实验

日期：2026-08-13

## 推荐项目名

**FastDeploy LLM Serving 在异构 GPU 上的推理生命周期与 KV Cache 实验设计**

短版：

**FastDeploy Serving Profiling on Heterogeneous GPUs**

## 一句话定位

围绕 Paddle FastDeploy LLM serving 栈，设计 NVIDIA RTX 4090、MetaX C500、Iluvatar MR-V100 三硬件对照实验，关注 request lifecycle、prefill/decode、KV cache capacity、prefix cache、chunked prefill、graph optimization 与 OpenAI-compatible serving 指标；当前已在 C500 上完成 readiness gate 和隔离源码硬装尝试，确认当前 Paddle 2.6/MACA 镜像只能推进到 Python 依赖 / import 诊断层，尚不具备真实 FastDeploy serving runtime。

## 当前稳妥可用版 Bullet

- 设计 FastDeploy LLM serving 跨硬件实验矩阵，覆盖 RTX 4090、MetaX C500、Iluvatar MR-V100 三类后端，统一测量 TTFT、TPOT、ITL、吞吐、p90/p99、KV cache block 数、cacheable tokens 与 OOM boundary。
- 将实验拆分为 readiness gate、offline `LLM.generate` smoke、OpenAI-compatible server smoke、prefill/decode shape sweep、KV cache capacity frontier、prefix cache mixed workload、chunked prefill mixed traffic、graph optimization 对照等阶段。
- 在 MetaX C500 上实现并运行 FastDeploy readiness probe，采集硬件可见性、Paddle package/device flags、FastDeploy/Paddle LLM serving 依赖、API server 依赖和 custom GPU op import 结果。
- 验证当前 C500 镜像硬件可见：MetaX C500 sGPU slice，25% compute / 16GB VRAM quota；Paddle 可用，版本为 `paddlepaddle-gpu 2.6.0+maca3.0.0.5`，设备路径为 `gpu:0`。
- 定位当前 C500 镜像不适合直接运行 FastDeploy LLM serving：缺少 `fastdeploy`、`paddlenlp/paddleformers`、`fastapi/uvicorn/openai`，且 `paddle.jit.marker.unified` 与 FastDeploy custom GPU op import 均失败。
- 通过非破坏性 package index 检查区分 NVIDIA CUDA wheel 与 MetaX 路线：当前可见 `fastdeploy-gpu==2.5.0` 为 CUDA 路线，未在当前 MACA 源看到 `fastdeploy-metax-gpu` / `paddle-metax-gpu` wheel。
- 在隔离 venv 中尝试 FastDeploy source hard-install，补齐 FastAPI/Uvicorn/OpenAI/Transformers/PaddleFormers 等 Python serving 依赖，并将导入链推进到 `fastdeploy.engine.args_utils`、`cache_manager` 和 `model_executor.ops.gpu` Python package。
- 进一步定位硬装的真实 blocker：FastDeploy `develop` 依赖 Paddle 新 API（`paddle.enable_compat`、`paddle.device.get_device_properties`），当前 Paddle 2.6 缺失；临时 shim 后 `LLM` / `SamplingParams` 可 import，但关键 GPU custom op `beam_search_softmax` 仍未编译/加载。
- 分析 build-path mismatch：当前 Paddle 暴露 `is_compiled_with_cuda=True` 和 `gpu:0`，但容器没有 `nvcc`；FastDeploy MetaX ops 需要 `metax_gpu` custom-device 路线，而当前镜像 `custom_device_types=[]`。

## 等真正跑通 FastDeploy Serving 后升级版 Bullet

- Built and profiled FastDeploy LLM serving across NVIDIA RTX 4090, MetaX C500, and Iluvatar MR-V100, measuring TTFT/TPOT/ITL, request throughput, output token throughput, KV-cache capacity, and p90/p99 tail latency under mixed prompt workloads.
- Quantified how `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, and `block_size` trade off context length, concurrency, and KV-cache capacity across heterogeneous Paddle backends.
- Evaluated prefix caching under 0/25/50/75/100% shared-prefix workloads, measuring TTFT improvement and tail-latency impact under realistic mixed traffic.
- Evaluated chunked prefill under mixed short/long prompt traffic, isolating how `max_num_batched_tokens` and `max_num_partial_prefills` affect short-request p99 latency and aggregate throughput.
- Compared graph optimization / CUDAGraph behavior across NVIDIA and Iluvatar paths, separating startup/warmup overhead from steady-state decode latency.

## 60 秒面试讲法

我想把 Paddle 经验往推理框架优化靠，所以没有停在 Paddle Inference demo，而是选了 Paddle 生态里更接近 vLLM/SGLang 的 FastDeploy。我的实验设计是三硬件对照：4090 做 NVIDIA baseline，C500 做 MetaX 兼容性目标，MR-V100 做 Iluvatar 异构后端。统一 workload 不是只测 tokens/s，而是拆 request lifecycle、prefill/decode、KV cache、prefix cache、chunked prefill 和 graph optimization。

目前我先在 C500 上跑了 readiness gate 和源码硬装尝试。结果是硬件和 Paddle 都可见，C500 是 25% compute / 16GB quota 的 sGPU slice，Paddle 2.6 走 `gpu:0`；Python serving 依赖可以在隔离 venv 里补齐，甚至可以通过 shim 让 `LLM` / `SamplingParams` import 通过。但真正 blocker 在 GPU custom ops 和后端分发：当前 Paddle 缺少 FastDeploy 预期的新 API，且没有 `beam_search_softmax` 这类已编译 custom op；同时构建逻辑会看到 CUDA-compatible Paddle，但容器没有 `nvcc`，而 MetaX ops 又要求 `metax_gpu` custom-device 路线。所以这个结果说明当前镜像适合 Paddle runtime / GraphNet 研究和 FastDeploy 兼容性诊断，但不适合直接产出 FastDeploy TTFT/TPOT serving 数据。

## 不要过度声明

不要写：

- 已经在 C500 上跑通 FastDeploy LLM serving
- 已经拿到 C500 FastDeploy TTFT / TPOT
- 已经证明 C500 FastDeploy prefix cache 或 chunked prefill 有收益
- 当前 C500 Paddle 2.6/MACA 镜像兼容 FastDeploy MetaX 路线
- Python-only hard-install 等价于 FastDeploy serving runtime

当前最强真实 claim：

- 完成 FastDeploy 推理框架实验设计
- 完成 C500 readiness gate 和软件栈缺口定位
- 完成 C500 隔离 source hard-install 尝试，定位到 Paddle API drift、CUDA-vs-MetaX backend dispatch mismatch、custom GPU ops 未编译三类 blocker
- 明确当前 C500 镜像不是 FastDeploy serving 镜像
- 给出下一次换镜像后的实验执行矩阵

## 证据索引

| 证据 | 文件 |
| --- | --- |
| FastDeploy 实验矩阵 | `configs/fastdeploy/experiment_matrix.json` |
| FastDeploy readiness 脚本 | `scripts/fastdeploy_readiness_probe.py` |
| C500 readiness 原始数据 | `raw/metax-c500-fastdeploy/fastdeploy_c500_readiness_probe.json` |
| C500 package index 原始日志 | `raw/metax-c500-fastdeploy/fastdeploy_package_index_probe.log` |
| C500 source hard-install 原始数据 | `raw/metax-c500-fastdeploy/hardinstall/` |
| 跨硬件实验设计 | `notes/fastdeploy-cross-hardware-experiment-design.md` |
| C500 readiness 报告 | `notes/fastdeploy-c500-readiness-report.md` |
| C500 hard-install 报告 | `notes/fastdeploy-c500-hardinstall-attempt.md` |
