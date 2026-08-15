# Quant Per Channel Cast Fused C500 Optimization

## 1. Case Summary

- 算子：`quant_per_channel_cast_fused`
- 语义：对每个 hidden channel 独立做 128-token block FP8 量化，并可选完成 token expand / gather 与 FP8 rescale
- 四条执行路径：
  - Plain
  - Expand
  - Rescale
  - RescaleExpand
- 支持场景：BF16、FP32、FP8 E4M3FN Rescale、`round_scale=True/False`、连续与非连续输入、边界规模和参数合法性检查
- 源实现：`MetaX-MACA/TileKernels-Metax::tile_kernels/quant/per_channel_cast_fused_kernel.py`
- 源提交：`f689bfb2062e451b6c784d0107f9fcd8180630b6`
- 被测提交：`0f0cad77a8f7e5af2dd3ec64fd0ee981fe235e1a`
- 目标硬件：MetaX C500

这个案例的核心是：同一个 fused cast 算子有四条路径，不能用单一 staging / tile 策略覆盖所有路径。Plain / Expand 主要是搬运与 absmax，适合 thread-local staging；Rescale / RescaleExpand 同时持有 FP8 输入、scale inverse 和 rescale 中间值，寄存器压力更高，因此保留 shared staging 更稳。

核心定位链路：

```text
upstream tile too large
-> FP32 staging tile can exceed C500 64 KiB shared limit
-> large tile also reduces workgroup count and parallelism
-> Plain/Expand still pay unnecessary shared round-trip
-> production kernel fixes tile_k=64 and splits staging policy by path
-> lower shared traffic, safe launch config, more workgroups, better memory-bound throughput
```

## 2. Baseline Problem

### 2.1 Shared memory footprint can exceed C500 limit

原始 TileKernels 参考实现采用较大的 hidden tile。FP32 路径中，`128 x 128` staging tile 加 reduction scratch 约需 `66 KiB` shared memory，超过 C500 单 workgroup `64 KiB` 上限，无法直接运行。

最终代码中明确写出这个约束：

```python
_TILE_K = 64
_C500_SHARED_MEMORY_LIMIT_BYTES = 64 * 1024
```

并用 `_shared_memory_bytes()` 在构造期提前检查，避免编译或 launch 后才失败。

### 2.2 Large tile reduces workgroup parallelism

较大的 hidden tile 会减少 `ceildiv(hidden, tile_k)` 的 workgroup 数量。在 C500 上，`tile_k=64` 一方面贴合 64-lane wavefront，另一方面让 hidden 维分出更多 workgroup，提高并行度。

### 2.3 Plain / Expand paths have unnecessary shared staging

Plain 和 Expand 路径没有额外 rescale scale 输入，输入值只需要在 absmax 归约后再用于 fp8 写出。原始路径把中间数据写到 shared 再读回，形成不必要的中间搬运。

最终策略：

```python
"register_staging": not self.with_rescale
```

也就是 Plain / Expand 使用 thread-local staging，Rescale / RescaleExpand 保留 shared staging。

### 2.4 Rescale paths need register-pressure control

Rescale 和 RescaleExpand 路径除了输入值，还要持有 `x_sf_invs`、source token 映射和 rescale 后逻辑值。若强行也改成 register staging，容易增加寄存器压力。最终实现保留 shared staging，以更稳地控制 occupancy 和 live values。

## 3. Optimization Ideas

### 3.1 Implement four explicit execution paths

Kernel 用两个布尔维度组合出四条路径：

```python
with_rescale: bool
with_expand: bool
```

对应：

| Path | `with_rescale` | `with_expand` | Meaning |
| --- | --- | --- | --- |
| Plain | False | False | 直接 per-channel FP8 cast |
| Expand | False | True | 先按 `pos_to_token` gather/expand，再 cast |
| Rescale | True | False | FP8 输入按原 scale inverse 还原逻辑值，再重新 cast |
| RescaleExpand | True | True | gather/expand + rescale + recast |

这样可以按路径选择 staging 和线程映射，而不是用一个折中策略覆盖所有场景。

### 3.2 Fix hidden tile to C500 wave-friendly `tile_k=64`

最终实现固定：

```python
_TILE_K = 64
```

意义：

- 适配 C500 64-lane wavefront。
- 避免 FP32 path shared memory 超过 64 KiB。
- 比 upstream 128/256-column tile 暴露更多 workgroups。
- 对所有支持路径都是 safe default。

### 3.3 Use local staging for Plain / Expand

当 `register_staging=True` 时：

```python
x_staging = T.alloc_local((vec_m, vec_k), in_dtype)
```

读取输入后，值保存在 thread-local staging 中；完成 absmax reduction 和 scale 计算后，再从 local 读出写 fp8：

```python
if register_staging:
    in_local[j] = x_staging[i, j]
else:
    in_local[j] = x_shared[m_id * vec_m + i, k_id * vec_k + j]
```

收益：

- Plain / Expand 减少一轮 shared-memory write/read。
- HBM 必要访问量基本保持不变，说明收益主要来自减少中间数据搬运。

### 3.4 Keep shared staging for Rescale / RescaleExpand

Rescale 路径默认：

```python
"register_staging": not self.with_rescale
```

即 `with_rescale=True` 时使用 shared staging：

```python
x_shared = T.alloc_shared((tile_m, tile_k), in_dtype)
```

这是一个寄存器压力与 shared traffic 的折中：Rescale live values 更多，保留 shared 可以避免 thread-local staging 把寄存器推太高。

### 3.5 Choose `threads_per_token` by path and output scale

固定总线程：

```python
_NUM_THREADS = 256
```

但每 token 分配的线程数按路径与规模选择：

```python
_PLAIN_THREADS_PER_TOKEN = 16
_SMALL_RESCALE_THREADS_PER_TOKEN = 8
_DEFAULT_RESCALE_THREADS_PER_TOKEN = 16
_LARGE_RESCALE_THREADS_PER_TOKEN = 32
```

选择逻辑：

```python
def _rescale_threads_per_token(num_tokens_out: int, with_expand: bool) -> int:
    if num_tokens_out <= 256:
        return 8
    if num_tokens_out >= 4096 or (with_expand and num_tokens_out >= 2048):
        return 32
    return 16
```

作用：

- 小输出规模降低每 token 并行度，避免浪费线程。
- 大输出规模提高每 token 并行度，增强访存连续性和吞吐。
- Expand 场景更早切到大并行度，因为 `pos_to_token` gather 增加了不规则性。

### 3.6 Validate configs early

最终构造函数检查：

- `threads_per_token` 必须是 `8/16/32/64`
- `tile_k` 必须能被 `threads_per_token` 整除
- shared memory footprint 不得超过 `64 KiB`

对应代码：

```python
if self.shared_memory_bytes > _C500_SHARED_MEMORY_LIMIT_BYTES:
    raise ValueError(...)
```

这避免了 C500 上“能编译、launch 才失败”或更危险的静默错误。

## 4. Final Kernel Structure

Kernel 的公共结构：

```text
grid: token block x hidden block
tile_m = 128 tokens
tile_k = 64 hidden columns
threads = 256
vec_k = tile_k // threads_per_token
vec_m = tile_m * threads_per_token // threads
```

每个 workgroup 做：

```text
optional load pos_to_token
optional load x_sf_invs
load input tile
optional rescale logical value
compute per-column absmax
reduce amax across token block
compute scale and inverse scale
write out_sf
reload staged input from local/shared
optional rescale
multiply by sf_inv
cast/write fp8 output
```

Scale 计算：

```python
clamped_amax = T.max(amax, _MIN_AMAX)
sf = clamped_amax / _FP8_MAX
sf_inv = _FP8_MAX / clamped_amax
```

`round_sf=True` 时通过 bit reinterpret 做 power-of-two scale rounding：

```python
bits = T.reinterpret(sf, T.uint32)
exp_sf = ((bits - 1) >> 23) + 1 - 127
sf = T.reinterpret((127 + exp_sf) << 23, T.float32)
sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
```

## 5. Correctness

参考实现：

- 独立 PyTorch Eager 实现，不复用 TileLang kernel 计算过程。
- TileKernels TileLang baseline 另做可信度验证。

测试命令：

```bash
git checkout 0f0cad77a8f7e5af2dd3ec64fd0ee981fe235e1a
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:$PYTHONPATH
python scripts/validate_manifest.py
python scripts/validate_manifest.py --check-op QuantPerChannelCastFusedOp --strict
python -m pytest -q tests/ops/test_per_channel_cast_fused.py
python -m pytest -q benchmarks/tests/test_per_channel_cast_fused_baseline.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
```

验证范围：

- Plain / Expand / Rescale / RescaleExpand
- BF16 / FP32 / FP8 E4M3FN
- `round_scale=True/False`
- 连续与非连续输入
- 边界规模
- 参数合法性检查
- PR A 原始 9 项 workload
- 当前 Manifest 20 项正式 workload

误差：

- FP8 输出逐元素精确比较
- Scale 输出 `atol=1e-7`, `rtol=1e-6`
- BF16 / FP32 按仓库规范

记录结果：

| Check | Result |
| --- | --- |
| Production Kernel correctness | 45/45 passed |
| TileKernels baseline credibility | 7/7 passed |
| PR A workloads | 9/9 passed |
| Benchmark related tests | 24 passed |
| Manifest tests | 7 passed |
| Current Manifest workloads | 20/20 runnable |

## 6. C500 Performance

环境：

| Item | Value |
| --- | --- |
| GPU | MetaX C500, 64 GiB, full GPU instance |
| GPU Driver | 3.8.30 |
| MACA | 3.7.1.5 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10 maca |
| Python | 3.12 |
| Commit | `0f0cad77a8f7e5af2dd3ec64fd0ee981fe235e1a` |

复现命令：

```bash
git checkout 0f0cad77a8f7e5af2dd3ec64fd0ee981fe235e1a
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:$PYTHONPATH
./scripts/run_quant_per_channel_cast_fused.sh benchmark
```

Benchmark 协议：

- 10 warmup
- 50 repeats
- 3 trials
- 取 3 轮平均执行时间的中位数
- JIT compile time 不计入
- 每次测量前 L2 cache flush
- 计时边界设备同步

整体结果：

- 相对 PyTorch Eager：20/20 胜出，几何平均 `4.1808x`
- 相对 TileKernels TileLang：20/20 胜出，几何平均 `2.7793x`

代表性 workload：

| Scenario | Production | TileKernels | Speedup |
| --- | ---: | ---: | ---: |
| Plain FP32, `1024x3072` | 0.0369 ms | 0.0878 ms | 2.3794x |
| Expand FP32, `1001x3072 -> 2048` | 0.0584 ms | 0.2546 ms | 4.3596x |
| Rescale FP8, `1024x3072` | 0.0630 ms | 0.1550 ms | 2.4603x |
| RescaleExpand FP8, `1001x3072 -> 2048` | 0.1182 ms | 0.3036 ms | 2.5685x |

mcProfiler / Roofline 结论：

- 主要瓶颈是数据搬运和访存开销。
- Plain / Expand 去掉 shared staging 后，shared load/store 指令数明显下降。
- 必要 HBM 访问量基本保持不变，说明收益来自减少中间数据搬运。
- mcProfiler 3.8.1.4 C500 全卡模型：最大显存带宽 `1843.2 GB/s`，ridge point `260 FLOP/Byte`。
- 20 项 workload 算术强度均在 ridge point 左侧，因此整体属于 memory bandwidth limited。

## 7. Transferability To RTX 4090

### Likely Portable

- 四路径拆分是通用方法：Plain/Expand 与 Rescale 类路径的 live values 和 staging 需求确实不同。
- 构造期配置校验通用，尤其是 shared memory footprint、tile divisibility 和 vector width。
- Plain/Expand 去除不必要 shared round-trip 通常也适用于 NVIDIA。
- 根据 output size / expand 여부选择线程映射的思想可迁移。
- 保留 PyTorch Eager 和 migrated TileLang 双 baseline 的 benchmark 设计很适合 4090 对照。

### Needs Rechecking On 4090

- `tile_k=64` 是 C500 64-lane wavefront 和 64 KiB shared 限制下的稳健选择。4090 warp 是 32 threads，shared limit、occupancy、L1/shared 配置不同，`tile_k=64/128` 都应 sweep。
- `_NUM_THREADS=256` 和 `threads_per_token in {8,16,32,64}` 需要重调；4090 可能偏好不同 block size。
- Rescale 路径是否仍应保留 shared staging，要看 4090 register count、occupancy 和 spill 情况。
- FP8 cast / rescale 在 NVIDIA 后端可能有不同 lowering，不能直接套 C500 的瓶颈归因。
- L2 flush、profiler 带宽模型和 roofline 分母需要用 NVIDIA 工具链重新定义。

### Suggested 4090 Experiment

建议至少准备四组版本：

1. `pytorch-eager`: 独立 PyTorch reference。
2. `tilekernels-port`: 迁移 baseline。
3. `c500-production`: 当前 production kernel 配置。
4. `4090-sweep`: sweep `tile_k`, `threads_per_token`, `register_staging`, `_NUM_THREADS`。

建议 sweep：

| Parameter | Values |
| --- | --- |
| `tile_k` | 64, 128 |
| `_NUM_THREADS` | 128, 256, 512 |
| Plain/Expand staging | local, shared |
| Rescale staging | local, shared |
| `threads_per_token` | 8, 16, 32, 64 |

Nsight Compute 关注：

- global load/store throughput
- shared load/store instructions
- shared bank conflicts
- local memory / spills
- register count
- achieved occupancy
- L2 hit rate
- instruction mix, especially conversion and integer ops for FP8 path

判定逻辑：

- 如果 Plain/Expand local staging 仍胜出，说明“减少 shared round-trip”是跨硬件通用收益。
- 如果 Rescale local staging 在 4090 胜出且无明显 spill，则 C500 的 shared-staging 保守策略可以做硬件分支。
- 如果 `tile_k=128` 在 4090 胜出，说明 C500 的 `tile_k=64` 主要来自 shared limit / wavefront / workgroup 数约束。
- 如果 TileKernels baseline 在 4090 不输 production，说明 C500 的大 tile / shared 限制是特定硬件约束。

## 8. Open Limits

- FP32 原始路径在 C500 上超过 shared memory，因此 baseline 曾对 FP32 `tile_k` 做能运行的最小调整；跨硬件比较时要明确 baseline 是否被修正。
- 当前实现没有手写 fp8 bit pack，仍依赖 TileLang / 后端 cast；FP8 path 在 4090 上需要单独检查 lowering。
- memory-bound 结论基于 mcProfiler 内置 C500 模型，不应直接搬到 4090。

## 9. Source Material

- PR / 网页记录：`/Users/yiweihan/.codex/attachments/942b3fc5-e7a9-4de4-9de8-ba0e039ea9c6/pasted-text.txt`
- 最终 kernel 源码快照：`/Users/yiweihan/Documents/muxi/doc/quant_per_channel_cast_fused_c500_kernel.py`
- 原始 kernel 附件：`/Users/yiweihan/.codex/attachments/30786005-02ca-48e2-bc46-7627415f0604/pasted-text.txt`

