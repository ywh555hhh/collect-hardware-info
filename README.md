# 收集硬件信息

这个仓库用于收集不同 AI 加速卡在真实租用环境里向开发者暴露出来的硬件、驱动、运行时和框架信息。

第一批数据来自一台沐曦 / MetaX C500 机器，重点不是跑通某个推理框架，而是回答一个更底层的问题：

> 这张卡到底把哪些参数、计数器、架构信息和运行时属性暴露给了开发者？

## 当前数据

### NVIDIA RTX 4090 D

采集时间：2026-08-11 / 2026-08-12

这批数据记录 TileLang / TileKernels-Metax 三个 FP8/MoE 相关算子在 RTX 4090 D 上的 profile-guided 优化尝试。重点不是硬件属性枚举，而是沉淀一组真实 kernel 优化闭环：C500 迁移思路、4090 上的 NSYS profiling、correctness-gated benchmark、最终 adaptive 默认策略和代码快照。

环境摘要：

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D |
| CUDA | 12.8 |
| Driver | 570.124.06 |
| NSYS | 2024.6.2 |
| NCU | 2025.1.0 |
| NCU counter 状态 | 被 `RmProfilingAdminOnly=1` 阻塞 |
| Repository | TileKernels-Metax |

最终三算子结果：

| 算子 | 通过 rows | Gmean BW | Peak BW | Gmean speedup |
| --- | ---: | ---: | ---: | ---: |
| `per_channel_cast_fused` | 96/96 | 1044.7 GB/s | 1474.8 GB/s | 1.0604x |
| `batched_transpose` | 84/84 | 904.7 GB/s | 932.7 GB/s | 1.0054x |
| `swiglu_forward_and_per_channel_cast_and_transpose` | 224/224 | 773.3 GB/s | 870.1 GB/s | 1.0051x |

关键观察：

- DeepSeek/TileKernels 原实现已经很强，单 kernel 局部调参空间有限。
- 最大可写结果来自 `per_channel_cast_fused` 的 MoE/topk expand 路径：profile-guided adaptive 默认策略达到 `~6.0%` gmean speedup 和 `~1.475 TB/s` peak bandwidth。
- `batched_transpose` 的 `block_k=8 + threads=512` 是 correctness 修复后的真实小收益，不应包装成大 SOTA。
- SwiGLU 的 `128x64` tile 只适合 transpose path + `num_per_tokens=32`，全局启用会伤害 no-transpose path。
- NCU counter 当前不可用，因此本轮可信证据是 NSYS kernel summary + benchmark JSONL + correctness/pass rows。

相关目录：

```text
notes/
  nvidia-rtx4090d-tilelang-profile-guided.md
  nvidia-rtx4090d-tilelang-per-channel-sweep.md
  nvidia-rtx4090d-tilelang-three-operator-push.md
  nvidia-rtx4090d-tilelang-sota-push.md

kernels/nvidia-rtx4090d/tilelang-profile-guided/
  *_4090_profile_guided_kernel.py

probes/nvidia-rtx4090d/tilelang-profile-guided/
  run_*.sh
  profile_*.py

raw/nvidia-rtx4090d/tilelang-profile-guided-20260811/
  profile_guided_adaptive_20260811/
  profiling_20260811/
  opt_sweep/
  three_ops_push_20260811/
  sota_push_20260811/
```

### MetaX C500

采集时间：2026-08-12

环境摘要：

| 项目 | 值 |
| --- | --- |
| GPU | MetaX C500 / MXC500 |
| 物理显存 | 64GB HBM2e |
| 当前实例形态 | sGPU slice |
| sGPU 配额 | 25% compute / 15.625GB |
| MACA | 3.5.3.20 |
| KMD driver | 3.8.30 |
| BIOS | 1.33.5.0 |
| `mx-smi` | 2.2.12 |
| PyTorch | 2.8.0+metax3.5.3.9 |

运行时和框架暴露的关键参数：

| 参数 | 值 | 来源 |
| --- | --- | --- |
| AP / SM-like 单元数 | 104 | `mcGetDeviceProperties`, PyTorch |
| warp size | 64 | MACA runtime, PyTorch, headers |
| L2 cache | 8MB | MACA runtime, PyTorch |
| shared memory per block | 64KB | MACA runtime, PyTorch |
| registers per block / multiprocessor | 131072 | MACA runtime, PyTorch |
| max threads per block | 1024 | MACA runtime |
| max threads per AP | 2048 | MACA runtime, PyTorch |
| memory bus width | 4096-bit | MACA runtime |
| xcore clock | 1125MHz | `mx-smi --show-clock` |
| memory controller clock | 1800MHz | `mx-smi --show-clock` |
| PCIe | Gen5 x16, 32 GT/s | `mx-smi --show-pcie` |
| ECC | Enabled | `mx-smi --show-ecc-state` |

重要观察：

- 当前租到的是 sGPU 切片，`mx-smi` 能看到物理卡 64GB，但 PyTorch / MACA runtime 只暴露约 15.6GB。
- C500 的 warp size 是 64，不是 NVIDIA 常见的 32；跨平台 kernel 里不能硬编码 32。
- `mx-smi --show-ap-usage` 按 `8 dpc x 13 ap` 展示 AP usage，和 runtime 暴露的 `multiProcessorCount = 104` 对齐。
- `mxcc` 可以编译并运行自定义 MACA kernel，最小 `add1` 探针已经通过。
- `mx-smi` 暴露的监控面比较丰富，包括 DPM level、clock、HBM/PCIe bandwidth、AP usage、VPU usage、ECC、功耗、电压、电流、温度和进程信息。

## 目录结构

```text
raw/
  nvidia-rtx4090d/
    tilelang-profile-guided-20260811/ # benchmark JSONL + NSYS stats + pytest logs
  metax-c500/
    c500_hardware_exposure.txt  # mx-smi help/plain/query + PyTorch props + MACA runtime props
    c500_mxsmi_detail.txt       # mx-smi 各 show 子命令详细输出

probes/
  nvidia-rtx4090d/
    tilelang-profile-guided/     # TileLang benchmark/profile reproduction scripts
  metax-c500/
    add1.cpp                    # 最小 MACA 自定义 kernel 编译/运行探针
    device_props.cpp            # mcGetDeviceProperties 属性采集探针

kernels/
  nvidia-rtx4090d/
    tilelang-profile-guided/     # final profile-guided TileLang kernel snapshots
```

## 复现方式

### 编译 MACA runtime 属性探针

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:$PATH

cd probes/metax-c500
mxcc -x maca -offload-arch native device_props.cpp -o device_props --maca-path=/opt/maca
./device_props
```

### 常用 `mx-smi` 采集命令

```bash
mx-smi
mx-smi -j
mx-smi -L
mx-smi --summary
mx-smi --show-hwinfo
mx-smi --show-version
mx-smi --show-memory
mx-smi --show-clock
mx-smi --show-clocks all
mx-smi --show-dpm all
mx-smi --show-dpm cur
mx-smi --show-core-usage
mx-smi --show-ap-usage
mx-smi --show-hbm-bandwidth
mx-smi --show-pcie
mx-smi --show-pcie-bandwidth
mx-smi --show-ecc-state
mx-smi --count-ecc
mx-smi --show-temperature
mx-smi --show-board-power
mx-smi --show-power-state
mx-smi --show-clk-tr
mx-smi --show-process
mx-smi --show-all-process
```

## 后续计划

- 为每张卡补一份统一格式的 `summary.md`。
- 添加内存带宽、kernel launch latency、cache/memory microbenchmark。
- 对比 C500、天数、Iluvatar、NVIDIA 等卡在 runtime/device props 上的差异。
- 记录对 LLM inference / tensor runtime / custom op 开发真正有用的硬件约束。
