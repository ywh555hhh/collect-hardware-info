# 简历项目稿：PaddlePaddle / C500 / GraphNet / 推理路径

日期：2026-08-12

## 推荐项目名

**PaddlePaddle GraphNet 计算图与推理路径在 MetaX C500 上的后端适配研究**

短版：

**Paddle GraphNet / Inference Bring-up on MetaX C500**

## 一句话定位

围绕非 NVIDIA 加速卡 MetaX C500，系统性验证 PaddlePaddle 后端暴露、算子覆盖、动态图/静态图、Paddle Inference Predictor 与官方 GraphNet 计算图样本的兼容性，定位 C500 Paddle 镜像在推理部署与 tensor compiler benchmark 方向的可用边界。

## 简历 Bullet：稳妥可用版

- 在 MetaX C500 sGPU 环境上构建 PaddlePaddle 后端能力探针，验证 `paddlepaddle-gpu 2.6.0+maca3.0.0.5` 通过 CUDA-compatible `gpu:0` 路径暴露设备，而非 Paddle custom device / `maca:0` 路径。
- 系统性验证 C500 Paddle 镜像的核心算子覆盖与延迟，包括 fp32/fp16 matmul、conv2d、softmax、layernorm、gather、scatter，并确认 Paddle Inference API 可用、`paddle.utils.run_check()` 通过。
- 实现 mini GraphNet-style 计算图 workload，覆盖 dense block、sparse gather/scatter block、mixed graph block 与 graph readout，比较 Paddle 动态图与 `paddle.jit.save/load` 静态图执行路径。
- 在 C500 上验证 dense/mixed graph block 的 `paddle.jit.save/load` 静态图执行，dense block 从 `0.333 ms` 动态图降低到 `0.255 ms` JIT 路径，mixed block 从 `1.268 ms` 降到 `1.122 ms`。
- 将固定形状 dense/mixed graph block 导出到 Paddle Inference，验证 `PaddleInferPredictor` 在 C500 上可执行，测得端到端 predictor latency 分别为 `1.515 ms` 与 `2.670 ms`。
- Bring up 官方 PaddlePaddle/GraphNet `ernie-3.0-nano-zh` Paddle sample：direct dygraph forward 在 `Place(gpu:0)` 上成功，输出 shape `[1, 312]`；进一步定位官方 benchmark static path 失败于 `paddle.jit.to_static` / generated `_C_ops` 兼容问题；将全部 159 个 generated `_C_ops` 降级为高层 Paddle API 后，official `compiler=nope` benchmark 跑通，eager/compiled e2e median 分别为 `4.415 ms` / `4.409 ms`。
- 识别官方 GraphNet 在 C500 Paddle 镜像上的适配 gap：上游 Paddle benchmark 存在 backend import path mismatch，且默认 `--device cuda` 与当前镜像要求的 `gpu/gpu:0` device string 不完全一致，需要 thin wrapper 或小补丁。
- 明确当前镜像能力边界：`is_compiled_with_cuda=True` 但 `is_compiled_with_cinn=False`，因此适合 runtime/operator/static graph/Paddle Inference 路径研究，不应声称完成 CINN speedup benchmark。

## 更像资深 Paddle / 推理工程师的讲法

这不是“我跑了一个 Paddle demo”，而是一次小型后端 bring-up：

1. 先识别框架如何看到硬件：device string、compiled flags、custom device、runtime library、sGPU quota。
2. 再拆算子覆盖：GEMM、conv、normalization、softmax、gather/scatter，对应推理中的 dense compute 与 irregular memory access。
3. 再走图执行链路：dynamic graph -> `paddle.jit.save/load` -> Paddle Inference Predictor。
4. 最后接官方 GraphNet：真实计算图 sample direct dygraph 能跑，static benchmark path 有可定位的兼容 gap。

面试时可以强调：我关心的不是单点 tokens/s，而是 framework/runtime/graph/predictor 的整条执行链路，以及非 NVIDIA 后端适配时常见的 device abstraction、operator signature、static graph conversion 问题。

## 60 秒面试版本

我用一台 MetaX C500 做了一个 PaddlePaddle 后端和推理路径的 bring-up 项目。第一步不是直接跑模型，而是确认 Paddle 怎么暴露这张卡：这个镜像是 `paddlepaddle-gpu 2.6.0+maca3.0.0.5`，设备走 CUDA-compatible `gpu:0`，不是 custom device，也没有 CINN。然后我做了核心算子 smoke，覆盖 matmul、conv、softmax、layernorm、gather、scatter。

之后我把它往 GraphNet 和推理路径上推：实现了 mini GraphNet-style workload，比较动态图、`paddle.jit.save/load` 静态图和 Paddle Inference Predictor。最后我还拉了官方 PaddlePaddle/GraphNet 的 `ernie-3.0-nano-zh` sample，在 C500 上 direct dygraph forward 成功，但官方 benchmark 的 static path 在 `to_static` 生成代码里失败。我把问题定位到 generated `_C_ops.full` 与 Paddle 2.6/MACA dy2static 的兼容问题，而不是简单的硬件不能跑。这个项目本质上是一次非 NVIDIA Paddle 后端的 runtime / graph / inference bring-up。

## 可被追问的技术点

| 追问方向 | 你应该怎么答 |
| --- | --- |
| 为什么 GraphNet 不是普通 GNN？ | 它是真实深度学习计算图数据集和 tensor compiler benchmark，用来评估编译器/后端优化能力。 |
| 为什么和推理有关？ | 推理部署依赖动态图、静态图导出、IR pass、Paddle Inference Predictor、算子覆盖和内存行为；GraphNet 给的是图级 workload。 |
| 为什么 `cuda` 不可用但 `gpu:0` 可用？ | Paddle Python device API 接受 `gpu/gpu:0`，但当前镜像底层仍是 CUDA-compatible 编译路径；这暴露了框架 device abstraction 与 benchmark 约定之间的 gap。 |
| 为什么 official benchmark failed 也有价值？ | 因为 direct dygraph sample 已经成功，失败被定位到 static conversion / generated `_C_ops.full` 兼容问题，说明这是可工程化推进的 bring-up gap。 |
| 能不能声称 CINN？ | 不能。当前镜像 `is_compiled_with_cinn=False`，只能说完成了 runtime/operator/static graph/Paddle Inference 路径研究，并为 CINN-enabled 镜像预留 benchmark 路线。 |

## 证据索引

| 证据 | 文件 |
| --- | --- |
| 后端/device 探针 | `scripts/paddle_backend_probe.py` / `raw/metax-c500-paddle/paddle_backend_probe.json` |
| device alias 探针 | `scripts/paddle_device_alias_probe.py` / `raw/metax-c500-paddle/device_alias/paddle_device_alias_probe.json` |
| 核心算子探针 | `scripts/paddle_op_probe.py` / `raw/metax-c500-paddle/paddle_op_probe.json` |
| mini GraphNet workload | `scripts/paddle_mini_graphnet_probe.py` / `raw/metax-c500-paddle/mini_graphnet/mini_graphnet_probe.json` |
| Paddle Inference predictor | `scripts/paddle_inference_graphnet_probe.py` / `raw/metax-c500-paddle/paddle_inference_graphnet/paddle_inference_graphnet_probe.json` |
| official GraphNet bring-up | `scripts/official_graphnet_c500_bringup_probe.py` / `raw/metax-c500-paddle/official_graphnet/official_graphnet_c500_bringup_probe.json` |
| official static patch 实验 | `scripts/official_graphnet_static_patch_probe.py` / `raw/metax-c500-paddle/official_graphnet_static_patch/official_graphnet_static_patch_probe.json` |
| `_C_ops` 签名探针 | `scripts/paddle_cops_signature_probe.py` / `raw/metax-c500-paddle/cops_signature/paddle_cops_signature_probe.json` |
| 技术报告 | `notes/metax-c500-paddle-backend-graphnet-probe.md` |
| 官方 GraphNet 对齐报告 | `notes/paddle-graphnet-official-alignment.md` |

## 不要过度声明

不要写：

- 完成了完整官方 GraphNet benchmark
- 完成了 CINN speedup benchmark
- 做了 kernel-level profiling 或 vendor profiler 分析
- 完成生产级推理优化

当前最强真实 claim：

- 完成 C500 Paddle 后端能力图谱
- 完成 mini GraphNet-style dynamic/static/Paddle Inference 路径验证
- 跑通 official GraphNet Paddle sample 的 direct dygraph forward
- 通过 generated `_C_ops` rewrite 跑通一个 official GraphNet `compiler=nope` benchmark timing

## 下一步最值钱

1. 做一个最小 patch：修 official GraphNet backend import path。
2. 加 device wrapper：把 benchmark 逻辑 device `cuda` 映射到 Paddle runtime device `gpu:0`。
3. 系统性替换或修复 generated `_C_ops` 的 dy2static 兼容问题，至少让 `ernie-3.0-nano-zh` official benchmark 的 `compiler=nope` 跑出 timing。
4. 如果平台提供 CINN-enabled Paddle 镜像，再做 `nope` vs `cinn` 对比。
5. 再加 vendor profiler / `mx-smi` 时间序列，分析 gather/scatter 与 static graph predictor overhead。
