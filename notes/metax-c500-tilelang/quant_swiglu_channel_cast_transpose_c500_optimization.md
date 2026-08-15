# Quant SwiGLU Channel Cast Transpose C500 Optimization

## 1. Case Summary

- 算子：`quant_swiglu_fwd_channel_cast_transpose`
- Kernel：`swiglu_forward_and_per_channel_cast_and_transpose`
- 语义：融合 SwiGLU forward、per-channel / per-token-block FP8 e4m3 量化、scale factor 输出，以及可选 transpose
- 输入：`x[num_tokens, hidden * 2]`，通常为 bf16
- 输出：
  - `without_transpose=True`: `out[num_tokens, hidden]`
  - `without_transpose=False`: `out[hidden, num_tokens]`
  - `out_sf[num_tokens // num_per_tokens, hidden]`
- 源实现：`MetaX-MACA/TileKernels-Metax::tile_kernels/quant/`
- 目标硬件：MetaX C500
- 迁移目标：TileOPs 的 Manifest / stateless Op / device Kernel / correctness test / independent Benchmark

这个案例和 `batched_transpose` 不同：它不是单纯 DRAM-bound 的重排算子，而是融合了非线性计算和 fp8 量化写出。最终瓶颈落在 C500 的 fp32 到 fp8 转换和 bit-pack 指令发射上。

核心定位链路：

```text
T.cast(fp32 -> fp8_e4m3fn)
-> C500 无硬件 fp32->fp8 转换
-> 工具链落到 __maca_cvt_float2_to_fp8x2 软件模拟
-> 每条 fp8x2 转换约 60 条指令，且含 fp64/分支路径
-> DRAM 利用率只有约 38%，Compute Instructions busy duty > 100%
-> 手写 bit-exact _pack_e4m3
-> 转换路径变成约 12 条 fp32/int bit ops，端到端显著加速
```

## 2. Baseline Problem

### 2.1 fp32 -> fp8 conversion becomes instruction-bound

C500 没有硬件 fp32 到 fp8 转换单元。迁移版如果直接使用 TileLang `T.cast` 写 fp8，会落到工具链软件模拟函数：

```text
__maca_cvt_float2_to_fp8x2
```

原材料记录：

- 每条 fp8 转换约需 60 条指令。
- DRAM 利用率约为峰值的 38%。
- 计算流水线占用率约 130%，说明不是内存先堵住，而是指令发射过载。

mcProfiler 进一步显示：

| Counter / Observation | Value |
| --- | ---: |
| Compute Instructions busy duty | 113.87% |
| Total instructions | 约 8.8 亿 |
| VALU / MTE instructions | 约 7.27 亿，占 82.5% |
| FMA_F32 | 约 1.28 亿，占 VALU 18% |
| fp32 conversion instructions | 约 1.12 亿，占 VALU 15% |
| pure integer bit ops | 约 4.93 亿，约 VALU 三分之二 |
| MMA + memory class instructions | 合计不足 1% |

结论：优化后的最终瓶颈仍然主要在 `_pack_e4m3` 的移位、掩码、位或等 bit 操作上，而不是矩阵乘或全局内存。

### 2.2 shared memory round-trip is unnecessary in non-transpose path

迁移版会把激活中间结果写入 shared memory，再读回来做 absmax 归约和 fp8 写出。对于 `without_transpose=True` 路径，这个 shared round-trip 并不必要，因为输出仍保持 `(num_tokens, hidden)` 布局，线程可以从读取、SwiGLU、amax 到 fp8 写出都把自己的 4x4 patch 保留在寄存器中。

### 2.3 row-strip read hurts DRAM locality

读取阶段原先采用 16 元素行条带映射，每条加载指令会跨 8 行数据，破坏 DRAM page locality。记录中的实测读带宽约 `0.95 TB/s`。

### 2.4 transposed path has shared bank conflicts

`num_per_tokens=128` 的转置路径需要 shared staging。逻辑 4x4 transpose write 会产生四路 bank conflict，需要通过 padding + swizzled physical layout 处理。

### 2.5 fixed 256 threads and fixed pipeline are too rigid

迁移版固定使用 256 threads，转置路径固定四级流水，`tile_x`、`tile_y`、`transpose_stages` 都不可调，无法适配不同 workload 几何。

## 3. Optimization Ideas

### 3.1 Hand-written bit-exact `_pack_e4m3`

最终代码中新增 `_pack_e4m3(val)`，用 fp32 / integer bit operations 直接生成 fp8 e4m3 bit pattern，替换 `T.cast` 的软件模拟路径。

关键特性：

- bit-exact match `torch.float8_e4m3fn` casting。
- 对 20 个丢弃 mantissa bits 做 round-to-nearest-even。
- 保留 `abs(val) < 2^-6` 的 subnormal 分支。
- scale 已保证 `|val| <= 448`，因此无需 saturation 和 NaN / overflow 分支。
- 替代约 60 ops per fp8x2 的软件模拟，变成约 12 条 fp32 / integer bit ops。

关键片段：

```python
bits = T.reinterpret(val, T.uint32)
sign = ((bits >> 31) & 1) << 7
ab = bits & 0x7FFFFFFF

lsb = (ab >> 20) & 1
r = ab + 0x7FFFF + lsb
exp8 = ((r >> 23) - 120) & 0xF
mant = (r >> 20) & 0x7
normal = sign | (exp8 << 3) | mant
```

subnormal 分支中特别处理 `T.round` 半远离零和 torch ties-to-even 之间的差异：

```python
w = T.reinterpret(ab, T.float32) * 512.0
m_away = T.cast(T.round(w), T.uint32)
w2 = w + w
w2_i = T.cast(w2, T.uint32)
tie = (w2 == T.cast(w2_i, T.float32)) & ((w2_i & 1) == 1)
d = m_away - T.if_then_else(tie, m_away & 1, T.uint32(0))
```

记录中的效果：

- isolated write-path speedup 约 `1.5x`
- 端到端性能提升到原来的约 `1.41x`

### 3.2 Disable fast math to preserve bit-exact correctness

最终 kernel 明确不启用 `TL_ENABLE_FAST_MATH`。原因是 fast math 会近似 `expf` 和 division，导致 bf16-rounded activation 在 rounding boundary 附近偏移一个 bf16 quantum，破坏与 torch reference 的逐位一致。

这个取舍很重要：

- 当前目标：`torch.equal` 逐位一致，`atol=0, rtol=0`
- 代价：牺牲一部分性能
- 后续空间：如果项目明确 fp8 容差，可以重新评估 fast math 或近似路径

### 3.3 4x4 register patch read

两条路径都把 `x` 读成 `4 x 4` register patches：

```python
TILE_K = 4
tmp_l = T.alloc_local((TILE_K,), in_dtype)
tmp_r = T.alloc_local((TILE_K,), in_dtype)
```

每个 thread 覆盖连续 4 行 x 连续 4 列。相比 row-strip mapping，每条 load 指令只触及 1 到 2 行，而不是跨 8 行。

记录中的效果：

- 读带宽从约 `0.95 TB/s` 提升到约 `1.27 TB/s`
- kernel 源码注释中记录 C500 上 4x4 patch load 可测到约 `1.3-1.5 TB/s`

### 3.4 Non-transpose path keeps activation tile in registers

`without_transpose=True` 时，最终 kernel 删除了 `act_shared` 中间缓冲。线程从读入 `x` 到写出 fp8 都保留自己的 patch：

```python
x_act_reg = T.alloc_local((n_patches * TILE_K, TILE_K), in_dtype)
amax_reg = T.alloc_local((TILE_K,), T.float32)
```

读取时直接完成 SwiGLU：

```python
val = val_l / (1 + T.exp(-val_l)) * val_r
x_act_reg[i_ * TILE_K + j, k] = T.cast(val, in_dtype)
```

同时折叠当前 patch 的 per-column absmax：

```python
amax_reg[k] = T.max(
    amax_reg[k],
    T.abs(T.float32(x_act_reg[i_ * TILE_K + j, k])))
```

收益：

- 删除 bf16 activation tile 在 shared memory 中的两次读写。
- 非转置四个 workload 总延迟下降约 `12.8%`。
- shared memory access efficiency 达到 100%，无冲突。

### 3.5 Transpose path uses padded and swizzled shared layout

`without_transpose=False` 时仍需要 shared memory 完成跨线程转置。最终代码对 `num_per_tokens=128` 使用更多 padding 和 swizzled layout：

```python
act_shared = T.alloc_shared(
    (
        TILE_Y,
        TILE_X + TILE_K * (2 if num_per_tokens == 128 else 1),
    ),
    in_dtype,
)

if num_per_tokens == 128:
    T.annotate_layout(
        {act_shared: tilelang.layout.make_swizzled_layout(act_shared)}
    )
```

记录中的效果：

- `num_per_tokens=128` 的三个 transposed workloads 上约 `9-13%` 更快。
- `num_per_tokens=32` 保持更便宜的单 padding unswizzled mapping，调优后更快。

### 3.6 Increase threads and expose tuning knobs

默认线程数从 256 提升到 512：

```python
num_threads = int(self.config.get("num_threads", 512))
```

效果：

- shared-memory phases 中每线程工作量减半。
- occupancy 提升。
- 10 个 manifest workloads 总和相对 faithful port 约 `1.42x`。

同时开放 tuning knobs：

- `num_threads`
- `tile_x`
- `tile_y`
- `transpose_stages`

并用 `_validate_tuning_config()` 把会静默写错的配置提前报错，包括：

- `tile_x % num_per_tokens == 0`
- `num_per_tokens % 4 == 0`
- `hidden % tile_y == 0`
- `tile_y % transpose_stages == 0`
- `num_threads % (tile_y // 4) == 0`
- `tile_x % (4 * thread_shared_step) == 0`
- 非转置路径需要 `thread_shared_step % num_split_blocks == 0`
- 非转置路径需要 `num_threads >= tile_y * num_split_blocks`

## 4. Final Kernel Structure

最终 kernel 分为两个路径。

### 4.1 without_transpose=True

```text
4x4 read gate/value from x
-> clamp optional
-> SwiGLU in fp32
-> cast activation back to input dtype in register
-> per-column amax in register
-> write partial amax to shared
-> reduce amax and compute sf / sf_inv
-> register activation * sf_inv
-> _pack_e4m3
-> scalar/coalesced fp8 output write
```

该路径没有 `act_shared` activation round-trip。

### 4.2 without_transpose=False

```text
4x4 read gate/value from x
-> clamp optional
-> SwiGLU in fp32
-> write transposed patch to padded/swizzled act_shared
-> staged copy into fragment
-> reduce_absmax over num_per_tokens
-> compute sf / sf_inv
-> _pack_e4m3
-> output write in transposed layout
```

该路径保留 shared staging，因为转置需要跨线程交换数据。

## 5. Correctness

参考实现：

```text
quant_swiglu_channel_cast_transpose_torch
```

测试命令：

```bash
cd /data/TileOPs-Metax
python -m pytest tests/ops/test_quant_swiglu_channel_cast_transpose.py -v
```

覆盖 shape：

| Shape / Params | Coverage |
| --- | --- |
| `[4096, 4096]`, `nt=128` | transpose and non-transpose |
| `[4096, 8192]`, `nt=32` | `round_sf=True`, `clamp=0.5`, transpose and non-transpose |
| `[128, 4096]`, `nt=128` | minimum token block |
| `[4096, 1152]`, `nt=128` | small hidden, hidden=576 |
| `[4096, 5120]`, `nt=32`, `clamp=10.0` | clamp path |
| `[4096, 6144]`, `nt=128`, `round_sf=True`, non-transpose | rounded scale path |
| `[8064, 8192]`, `nt=128`, non-transpose | large token path |

异常测试：

- invalid shape
- invalid dtype
- invalid params

误差：`atol=0, rtol=0`，FP8 output 和 scale 均与 torch reference 逐位一致，通过 `torch.equal`。

记录结果：全部测试通过。

## 6. C500 Performance

环境：

| Item | Value |
| --- | --- |
| GPU | MetaX C500 |
| Driver | 3.8.30 |
| MACA | 3.7.1.5 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10+cuda.gitf549117c |

复现命令：

```bash
cd /data/TileOPs-Metax
python -m pytest benchmarks/ops/bench_quant_swiglu_channel_cast_transpose.py -vvs
```

Benchmark 协议：

- 10 warmup
- 3 trials
- 每 trial 50 samples
- 测量前后 GPU synchronize
- 不包含首次编译时间

优化后 vs torch reference：

| workload | TileOps ms | torch-ref ms | speedup |
| --- | ---: | ---: | ---: |
| `t4096-h2048-nt128` | 0.0725 | 1.0628 | 14.66x |
| `t4096-h576-nt128` | 0.0267 | 0.3760 | 14.08x |
| `t8064-h4096-nt128` | 0.2360 | 2.8032 | 11.88x |
| `t4096-h7168-nt32-clamp` | 0.2312 | 3.9759 | 17.20x |
| `t8064-h6144-nt32-round-clamp` | 0.3305 | 4.9904 | 15.10x |
| `t4096-h3072-nt128-round` | 0.1064 | 1.5550 | 14.61x |
| `t8064-h2560-nt128-round` | 0.1576 | 1.8216 | 11.56x |
| `t4096-h2560-nt32-clamp10` | 0.0881 | 1.4683 | 16.67x |
| `t8064-h3072-nt32-clamp10` | 0.1720 | 2.5138 | 14.62x |
| `t4096-h4096-nt32-round-clamp10` | 0.1342 | 2.3077 | 17.20x |

PR comment 中还记录了相对“初始迁移版”的收益：

| workload | TileOps ms | initial port ms | torch ms | vs initial | vs torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t4096-h2048-nt128` | 0.0870 | 0.1580 | 1.0375 | 1.82x | 11.93x |
| `t4096-h576-nt128` | 0.0288 | 0.0474 | 0.3542 | 1.65x | 12.31x |
| `t8064-h4096-nt128` | 0.3055 | 0.5578 | 2.7783 | 1.83x | 9.09x |
| `t4096-h7168-nt32-clamp` | 0.3004 | 0.4004 | 3.9607 | 1.33x | 13.18x |
| `t8064-h6144-nt32-round-clamp` | 0.4248 | 0.8187 | 4.9655 | 1.93x | 11.69x |
| `t4096-h3072-nt128-round` | 0.1271 | 0.2133 | 1.5347 | 1.68x | 12.08x |
| `t8064-h2560-nt128-round` | 0.1989 | 0.3658 | 1.7973 | 1.84x | 9.03x |
| `t4096-h2560-nt32-clamp10` | 0.1163 | 0.1588 | 1.4471 | 1.37x | 12.45x |
| `t8064-h3072-nt32-clamp10` | 0.2230 | 0.4250 | 2.4887 | 1.91x | 11.16x |
| `t4096-h4096-nt32-round-clamp10` | 0.1721 | 0.2371 | 2.2838 | 1.38x | 13.27x |

## 7. Transferability To RTX 4090

### Likely Portable

- 融合 SwiGLU、amax、scale、fp8 output，减少 kernel launch 和中间 tensor traffic。
- 4x4 register patch read 改善 DRAM locality。
- 非转置路径把 activation tile 保留在寄存器中，去掉不必要 shared round-trip。
- 将 tile、threads、transpose stages 做成可配置，并在构造期校验非法组合。
- bit-exact fp8 pack 的测试方法和边界处理方式可复用。

### Needs Rechecking On 4090

- NVIDIA Ada / RTX 4090 对 fp8 支持和编译器 lowering 与 C500 不同。`T.cast(fp32 -> fp8)` 是否仍走高成本软件路径，需要看 PTX/SASS 和 Nsight Compute。
- `_pack_e4m3` 在 4090 上可能不比 CUDA 工具链转换更快。如果 NVIDIA 后端已有高效转换，手写 bit ops 可能增加指令压力。
- 512 threads 是 C500 调优结果。4090 的 warp=32、SM occupancy、register file、shared memory 限制不同，需要重新 sweep 256/512/1024 等组合。
- `make_swizzled_layout` 对 4090 shared bank conflict 的收益需要用 Nsight Compute 验证。NVIDIA 也是 32 banks，但 transaction 分解和 conflict 指标不同。
- fast math 的精度/性能 tradeoff 需要重新评估。如果目标从 bit-exact 改成 fp8 容差，4090 上可能有更大的近似优化空间。

### Suggested 4090 Experiment

至少保留四个版本：

1. `torch-ref`: PyTorch reference。
2. `faithful-port`: 原始迁移版，保留 `T.cast`、shared round-trip、256 threads。
3. `pack-only`: 只替换 `_pack_e4m3`。
4. `full-optimized`: 手写 pack + 4x4 patch + register-resident non-transpose + swizzled transpose + 512 threads。

建议指标：

- latency / speedup
- achieved bandwidth
- instruction count
- integer instruction count
- fp32 instruction count
- local memory / spill load-store
- shared bank conflicts
- global load/store efficiency
- register count
- achieved occupancy

判定逻辑：

- 如果 `pack-only` 在 4090 上不涨反降，说明 C500 的 fp8 software conversion 是硬件/工具链特有瓶颈。
- 如果 `4x4 patch` 和 register-resident non-transpose 仍提升，说明访存路径重构是通用收益。
- 如果 full-optimized 的瓶颈仍在 integer bit ops，说明手写 pack 虽快于 C500 工具链，但也把最终上限锁在指令发射。
- 如果 fast math 可接受且显著加速，后续可做 `bit-exact` 和 `relaxed-fp8` 两条实现策略。

## 8. Open Limits

- 当前 bit-exact 目标可能牺牲部分性能；项目需要明确 fp8 量化输出的容差标准。
- `_pack_e4m3` 假设 scale 保证 `|val| <= 448`，因此省略 saturation 和 NaN/overflow 分支。
- transposed path 仍有 shared staging，优化重点是降低 conflict 和控制 register pressure。
- C500 最终瓶颈已经转向 bit-pack integer ops，继续做访存优化可能收益有限。

## 9. Source Material

- PR / 网页记录：`/Users/yiweihan/.codex/attachments/01055e94-9881-49f3-a2c9-bf243c2e5aed/pasted-text.txt`
- 最终 kernel 源码快照：`/Users/yiweihan/Documents/muxi/doc/quant_swiglu_channel_cast_transpose_c500_kernel.py`
- 原始 kernel 附件：`/Users/yiweihan/.codex/attachments/dc5c281e-185d-4a70-889c-ae1919c1397d/pasted-text.txt`

