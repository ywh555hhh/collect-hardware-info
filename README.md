# 收集硬件信息

这个仓库用于收集不同 AI 加速卡在真实租用环境里向开发者暴露出来的硬件、驱动、运行时和框架信息。

第一批数据来自一台沐曦 / MetaX C500 机器，重点不是跑通某个推理框架，而是回答一个更底层的问题：

> 这张卡到底把哪些参数、计数器、架构信息和运行时属性暴露给了开发者？

## 当前数据

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
  metax-c500/
    c500_hardware_exposure.txt  # mx-smi help/plain/query + PyTorch props + MACA runtime props
    c500_mxsmi_detail.txt       # mx-smi 各 show 子命令详细输出
  metax-c500-vllm/
    c500_lfm25_lifecycle.json   # LFM2.5 vLLM offline lifecycle
    kv_sweep_lfm25/             # LFM2.5 KV cache / concurrency sweep raw data
    kv_sweep_qwen3_0p6b/        # Qwen3-0.6B pure-attention baseline sweep
    api_bench_lfm25/            # LFM2.5 warm API server concurrency smoke
    api_bench_lfm25_warm/       # LFM2.5 eager warm steady-state API benchmark
    api_bench_lfm25_non_eager_warm/
                                  # LFM2.5 torch.compile / CUDAGraph warm API benchmark
    api_bench_lfm25_stream/       # LFM2.5 streaming TTFT / TPOT benchmark
    api_bench_qwen3_stream/       # Qwen3-0.6B streaming TTFT / TPOT baseline
    prefix_cache_qwen3/           # Qwen3-0.6B prefix cache / mixed prompt workload
    non_eager_probe/            # LFM2.5 torch.compile / CUDAGraph probe

probes/
  metax-c500/
    add1.cpp                    # 最小 MACA 自定义 kernel 编译/运行探针
    device_props.cpp            # mcGetDeviceProperties 属性采集探针

scripts/
  vllm_lifecycle_probe.py       # 采集 vLLM 模型生命周期
  vllm_kv_sweep.py              # 扫 max_model_len / gpu_memory_utilization
  vllm_api_bench.py             # OpenAI-compatible API server 并发 smoke
  vllm_prefix_cache_probe.py    # Prefix cache / mixed prompt workload 探针
```

## vLLM 推理实验

当前已经补了三类和推理优化更相关的数据：

- lifecycle：拆 `download/config/tokenizer -> model load -> backend selection -> KV cache -> warmup -> request`。
- KV/cache sweep：扫 `max_model_len` 和 `gpu_memory_utilization`，观察 cache tokens 和 max concurrency。
- API steady-state：常驻 OpenAI-compatible server，比较 eager 和 non-eager，并扫 prompt length。
- Streaming TTFT/TPOT：拆 first token 和 per-output-token latency，对比 LFM2.5 / Qwen3。
- Prefix cache：对比 shared-prefix 和 unique-prefix mixed workload，观察 TTFT 和吞吐变化。

入口文档：

- `notes/metax-c500-vllm-lifecycle.md`
- `notes/metax-c500-vllm-lifecycle-and-kv-cache.md`
- `notes/metax-c500-vllm-deeper-exploration.md`
- `notes/metax-c500-vllm-api-steady-state.md`
- `notes/metax-c500-vllm-streaming-ttft-tpot.md`
- `notes/metax-c500-vllm-prefix-cache-mixed-workload.md`

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
