# Batched Transpose C500 Optimization

## 1. Case Summary

- 算子：`batched_transpose` / `BatchedTransposeFwdOp`
- 语义：rank-3 tensor 后两轴转置，`[B, M, N] -> [B, N, M]`
- 典型场景：MoE 专家并行中的 `[num_experts, num_tokens, hidden]` 批量转置
- 源实现：`MetaX-MACA/TileKernels-Metax::tile_kernels/transpose/batched_transpose_kernel.py`
- 迁移目标：TileOPs 的 Manifest / Op / Kernel / Test / Benchmark 最小闭环
- 目标硬件：MetaX C500
- 算子类型：纯数据搬运，`FLOPs = 0`，主要评价指标是有效带宽和访存计数器

这个案例的核心价值不是某个固定 tile 配置，而是一个清晰的定位链路：

```text
runtime index 访问 thread-local register tile
-> C500 后端将局部数组落到 scratch/private memory
-> 纯 transpose kernel 出现大量 private read/write
-> 带宽卡在 SM 流式上限约 48-58%
-> 改成标量 staging + 编译期常量 scatter
-> private traffic 归零，算子进入 DRAM-bound 区间
```

## 2. Baseline Problem

迁移过来的参考实现使用一个 `block_k x block_k` 的线程局部寄存器子块，再配合 shared memory 做转置和 swizzle。问题出现在类似下面的访问模式：

```python
tmp[swizzle_j, k]
```

其中 `swizzle_j` 由 `tid` 推导，是运行期值。寄存器本身不可寻址，一旦线程局部数组存在运行期下标，编译器需要给这个数组分配真实地址。在 C500 上，这个数组落到了 scratch/private memory。

mcProfiler 在纯搬运算子里观测到：

| Metric | Baseline |
| --- | ---: |
| Private Read Instructions | 3,096,576 |
| Private Write Instructions | 6,967,296 |
| instruction throughput efficiency | 9.39% |
| average cycles per instruction | 4.0 |
| VL1 Hit Rate | 88.08% |

private read 的数量可从源码推导：

```text
64,512 wavefronts * 8 uint2 loads * 6 launches = 3,096,576
```

这说明 private traffic 不是相关现象，而是由那一处 runtime-indexed thread-local array 直接造成。

原材料还指出：同样写法在 NVIDIA 上可能不吃亏，`ptxas` 会将 local depot 优化回寄存器 select tree，SASS 中没有 `LDL/STL`。因此这个问题很可能是 C500 后端 lowering / private memory 路径暴露出来的，而不是 TileLang 语义层面的必然问题。

## 3. Optimization Ideas

### 3.1 删除寄存器子块，改成标量 staging

最终 kernel 不再用 runtime index 访问一个二维 thread-local tile，而是每个 lane 只把连续的 `read_vec` 个元素读入一维 local fragment：

```python
staged = T.alloc_local((read_vec,), dtype)

for k in T.vectorized(read_vec):
    staged[k] = x[pb, px * block_x + i, py * block_y + col * read_vec + k]

for k in T.unroll(read_vec):
    shared[col * read_vec + k, i ^ swizzle_mask] = staged[k]
```

关键点：

- `k` 通过 `T.unroll(read_vec)` 成为编译期常量。
- scatter 到 shared 的行号 `col * read_vec + k` 是静态展开出来的。
- 中间值保留在寄存器中，不再需要 scratch/private memory。

收益：bf16 / fp32 主瓶颈被解除，private read/write 归零。

### 3.2 只加宽全局读，不加宽写

fp8 的单元素只有 1 byte。如果每个 lane 标量读，wavefront 连续区间只有 `64 B`，低于一个 cache line。最终实现让每个 lane 读 `read_vec` 个连续元素，并把 `READ_BYTES_PER_LANE` 设为 `8`：

```python
READ_BYTES_PER_LANE = 8
read_vec = max(1, READ_BYTES_PER_LANE // elem_bytes)
```

因此每个 wavefront 的连续读区间可以扩到：

```text
64 lanes * read_vec * elem_bytes = 64 * 8 B = 512 B
```

这一步只加宽 global read。写回 output 时仍保持标量 store。原材料中的写宽 sweep 显示，写侧宽度从 `1/2/4/8/16` 增大时带宽严格变差：

```text
1.22 -> 1.10 -> 0.67 -> 0.35 -> 0.17 TB/s
```

结论：读宽和 shared staging 解耦，写侧保持标量，是这个 kernel 的关键结构。

### 3.3 按字节而不是按元素分配线程负载

不同 dtype 的元素字节数不同：

| dtype | elem bytes | 64 bytes/thread 对应元素数 |
| --- | ---: | ---: |
| fp8 | 1 | 64 |
| bf16 | 2 | 32 |
| fp32 | 4 | 16 |

最终实现把每线程 tile 负载统一成字节预算：

```python
TARGET_BYTES_PER_THREAD = 64
MAX_BYTES_PER_THREAD = 64
```

这比“每线程固定元素数”更适合跨 dtype 的纯搬运算子。C500 上超过 64 B/thread 后出现断崖：

- 128 B/thread: fp8 约 `-40%`
- 128 B/thread: bf16 约 `-18%`
- fp32 基本不受影响，但为了统一配置约束仍采用 64 B/thread 上限

### 3.4 shared tile 从 padding 改成 XOR swizzle

padding 在这个访问模式下有数学上界：一次 staging step 中，冲突组里的 32 个 lane 写同一列、32 个不同行。bank 差异只来自行跨距，而行跨距恒为：

```text
read_vec * width * elem_bytes / bank_bytes
= 8 * width / 4
= 2 * width
```

它永远是偶数，因此最多只能覆盖一半 bank。实测 padding 版本 shared memory access efficiency 为 `66.67%`。

最终实现改成：

```python
swizzle_step = max(1, 4 // elem_bytes)
mask(r) = (r // read_vec) * swizzle_step
shared[r, c ^ mask(r)]
```

代码中对应：

```python
swizzle_mask = col * swizzle_step % block_x
shared[col * read_vec + k, i ^ swizzle_mask] = staged[k]

out[...] = shared[i, j ^ ((i // read_vec) * swizzle_step % block_x)]
```

效果：

| Metric | padding | XOR swizzle |
| --- | ---: | ---: |
| shared memory access efficiency | 66.67% | 100.0% |
| average conflict cycles per instruction | 1.0 | 0.0 |
| average cycles per instruction | 3.0 | 2.0 |
| instruction throughput efficiency | 15.87% | 18.12% |

需要注意：作者验证过，为了满足所有 conflict-free 条件而牺牲读宽、block 数或 occupancy，很多 shape 反而变慢。因此这些条件更适合作为 profiler 诊断规则，不应机械地当成 tile selection 规则。

## 4. Final Kernel Structure

最终 kernel 的结构可以压缩成三段：

```python
with T.Kernel(shape_y // block_y, shape_x // block_x, batches,
              threads=threads) as (py, px, pb):
    shared = T.alloc_shared((block_y, block_x), dtype)
    tid = T.get_thread_binding()
    row, col = tid // row_threads, tid % row_threads
    swizzle_mask = col * swizzle_step % block_x
    staged = T.alloc_local((read_vec,), dtype)

    # 1. coalesced global read -> register staging
    # 2. static-index scatter -> swizzled shared tile
    for step in T.unroll(steps):
        i = step * rows_per_step + row
        for k in T.vectorized(read_vec):
            staged[k] = x[pb, px * block_x + i,
                          py * block_y + col * read_vec + k]
        for k in T.unroll(read_vec):
            shared[col * read_vec + k, i ^ swizzle_mask] = staged[k]

    T.sync_threads()

    # 3. unswizzle shared -> scalar coalesced output store
    for i, j in T.Parallel(block_y, block_x, loop_layout=layout):
        out[pb, py * block_y + i, px * block_x + j] = \
            shared[i, j ^ ((i // read_vec) * swizzle_step % block_x)]
```

配置由常量推导，而不是按 shape 查表：

| Constant | Value | Meaning |
| --- | ---: | --- |
| `TARGET_BYTES_PER_THREAD` | 64 | 每线程搬运负载目标 |
| `MAX_BYTES_PER_THREAD` | 64 | C500 上的性能断崖门禁 |
| `READ_BYTES_PER_LANE` | 8 | 每 lane 全局读宽度目标 |
| `TILE_ROW_BYTES` | 256 | tile 行覆盖的连续输入字节目标 |
| `SHARED_MEMORY_BYTES` | 64 KiB | C500 单 block shared memory 上限 |
| `SMALL_TILE_COUNT` | 32 | 小 workload 分支阈值 |

实现中的 `_check_config()` 明确提前检查：

- `block_y % read_vec == 0`
- `threads % (block_y // read_vec) == 0`
- `block_x % rows_per_step == 0`
- `bytes_per_thread <= 64`
- `block_x` 是 2 的幂，保证 XOR swizzle 不越界
- `block_x * block_y * elem_bytes <= 64 KiB`

## 5. Correctness

参考实现：

```python
torch.transpose(x, 1, 2).contiguous()
```

覆盖用例：

| Case | Shape | dtype | Purpose |
| --- | --- | --- | --- |
| minimum-bf16 | `[1, 64, 64]` | bf16 | 最小合法 tile |
| experts8-fp8 | `[8, 128, 64]` | fp8_e4m3 | fp8 通路 |
| experts32-fp32 | `[32, 64, 128]` | fp32 | fp32 通路和 batch=32 |
| nontrivial-bf16 | `[8, 192, 256]` | bf16 | 非平凡小 shape |

异常与 layout 门禁：

- rank 不是 3 时抛 `ValueError`
- `M` 或 `N` 不是 64 倍数时抛 `ValueError`
- fp16 不支持
- 行切片视图必须拒绝
- 列切片视图允许，并保持正确

误差：`rtol=0, atol=0`。这是纯搬运算子，结果必须逐位一致。

记录结果：

```text
python -m pytest tests/ops/test_batched_transpose.py -q
6 passed

scripts/validate_manifest.py
All manifest checks passed
```

## 6. C500 Performance

环境：

| Item | Value |
| --- | --- |
| GPU | MetaX C500, 104 SM, 65536 MiB, sGPU-M disabled |
| Driver | Kernel Mode Driver 3.8.30 |
| MACA | 3.7.1.5 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10+cuda.gitf549117c |
| Compiler | mxcc 1.0.0 |

方法：

- baseline、本实现、Torch 在同一进程中交错计时
- `do_bench(backend="cupti")`
- 10 warmup + 50 repeats x 3 trials，取中位数
- 带宽达成率分母为同 session elementwise kernel 测得的 SM 流式上限 `1.520 TB/s`

代表性结果：

| shape | dtype | baseline ms | optimized ms | torch ms | optimized TB/s | achieved | vs baseline | vs torch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[8, 4032, 2048]` | fp8 | 0.1596 | 0.1030 | 2.7013 | 1.283 | 84.4% | 1.55x | 26.2x |
| `[8, 4032, 2048]` | bf16 | 0.3334 | 0.1820 | 0.4058 | 1.452 | 95.6% | 1.83x | 2.23x |
| `[8, 4032, 2048]` | fp32 | 0.7067 | 0.3729 | 0.4758 | 1.417 | 93.3% | 1.90x | 1.28x |
| `[32, 8064, 4096]` | fp8 | 2.4321 | 1.6159 | 97.7787 | 1.308 | 86.1% | 1.51x | 60.5x |
| `[32, 8064, 4096]` | bf16 | 5.7816 | 3.1753 | 4.4976 | 1.331 | 87.6% | 1.82x | 1.42x |
| `[32, 8064, 4096]` | fp32 | 11.4794 | 6.0887 | 7.7983 | 1.389 | 91.4% | 1.89x | 1.28x |
| `[8, 4096, 4096]` | fp8 | 0.3065 | 0.2151 | 4.3301 | 1.248 | 82.1% | 1.42x | 20.1x |
| `[8, 4096, 4096]` | bf16 | 0.6893 | 0.3579 | 0.4246 | 1.500 | 98.7% | 1.93x | 1.19x |
| `[8, 4096, 4096]` | fp32 | 1.4218 | 0.7052 | 0.9472 | 1.523 | 100.2% | 2.02x | 1.34x |

结论：

- 相对迁移基线：`1.42x - 2.02x`
- 带宽达成率：从基线约 `48-58%` 提升到 `82-100%`
- bf16 / fp32 已基本进入 DRAM-bound 区间
- fp8 对 Torch 的高倍数主要来自 Torch 缺少 fp8 快速路径，不应单独用作 kernel 强度论据

mcProfiler 关键计数器：

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Private Read Instructions | 3,096,576 | 0 |
| Private Write Instructions | 6,967,296 | 0 |
| shared memory access efficiency | 66.67% | 100.0% |
| average conflict cycles per instruction | 1.0 | 0.0 |
| average cycles per instruction | 4.0 | 2.0 |
| instruction throughput efficiency | 9.39% | 18.12% |
| VL1 Hit Rate | 88.08% | 95.97% |

## 7. Transferability To RTX 4090

这个方案拆开看，有些思路大概率可迁移，有些可能是 C500 特有。

### Likely Portable

- 纯搬运算子按 bytes 而不是 FLOPs 建模。
- 避免 thread-local array 的 runtime index，尽量让局部 fragment 下标在 unroll 后成为编译期常量。
- 对 fp8 / int8 等小元素 dtype，加宽 global read 以保证每个 warp/wavefront 有足够连续字节。
- 将 global read width、shared staging layout、output store width 解耦，分别 sweep。
- 用 profiler 计数器证明“源码行 -> 编译产物 -> 硬件事件”的因果链，而不是只看端到端耗时。

### C500-Specific Or Needs Rechecking

- C500 的 wavefront 是 64 lanes；4090 warp 是 32 threads。`64 lanes * 8 B = 512 B` 的读区间推导需要改成 `32 threads * bytes_per_thread`。
- C500 的 shared bank 模型在文档中按 `32 banks * 4 B` 推导；NVIDIA 4090 也是 32 banks，但 bank conflict 行为、transaction 合并和 profiler 指标要用 Nsight Compute 重新验证。
- C500 上 runtime-indexed local tile 明确落 scratch；4090 上 ptxas 可能把它优化回寄存器 select tree，因此第 1 步在 4090 上可能收益很小，甚至只是代码风格收益。
- `MAX_BYTES_PER_THREAD = 64` 是 C500 sweep 得出的断崖；4090 需要重扫，不能直接套。
- `TL_DISABLE_WARP_SPECIALIZED`、TileLang lowering、CUDA target codegen 可能改变寄存器、local memory 和 occupancy。

### 4090 Experiment Plan

建议至少跑三组版本：

1. `baseline-register-tile`: 保留 runtime-indexed local tile 的迁移基线。
2. `static-staging`: 只做标量/向量 staging + 常量下标 scatter，不做 XOR swizzle。
3. `static-staging-xor-swizzle`: 最终 C500 结构。

每组覆盖 dtype 和 shape：

| Shape | dtype | Purpose |
| --- | --- | --- |
| `[8, 4032, 2048]` | fp8 / bf16 / fp32 | 非 2 幂 token axis，对齐压力更真实 |
| `[8, 4096, 4096]` | fp8 / bf16 / fp32 | 2 幂大 shape，观察最佳带宽 |
| `[8, 192, 256]` | bf16 / fp32 | 小 shape，区分 launch-bound 和 bandwidth-bound |

4090 上建议收集：

- runtime ms
- effective bandwidth
- achieved / measured streaming bandwidth
- local memory load/store transactions 或 local memory bytes
- register count
- spill stores / spill loads
- shared load/store bank conflicts
- L1/TEX 和 L2 hit rate
- occupancy / active warps

Nsight Compute 指标可先从这些类别找：

```text
Memory Workload Analysis
Source Counters
Scheduler Statistics
Launch Statistics
Occupancy
Shared Memory
```

判定逻辑：

- 如果 baseline 在 4090 上没有 local/spill traffic，而 C500 有，则“消 scratch”是 C500 后端特异收益。
- 如果 static staging 仍然比 baseline 快，说明即使 ptxas 能优化，显式 staging 仍改善了调度、寄存器或访存合并。
- 如果 XOR swizzle 在 4090 上减少 bank conflicts 但端到端不涨，说明算子已 DRAM-bound，和 C500 结论一致。
- 如果写侧 vectorization 在 4090 上变快，说明“写保持标量”是 C500 特定结论，4090 应单独分支。

## 8. Open Limits

- 当前实现要求 `M` 和 `N` 都是 64 的倍数，非对齐 shape 需要尾块或 fallback。
- fp8 这里只作为 1-byte container 搬运，不涉及 fp8 数值计算。
- 小 shape 受 launch overhead 主导，带宽数字不能代表 steady-state memory efficiency。
- C500 经验上限 `1.520 TB/s` 是同 session 测得的 stream reference，不是硬件理论峰值；超过 100% 属测量参照误差范围。

## 9. Source Material

- PR / 网页记录：`/Users/yiweihan/.codex/attachments/4e23ec36-b9c4-42a1-b98c-b5ce162d03c1/pasted-text.txt`
- 最终 kernel 源码快照：`/Users/yiweihan/Documents/muxi/doc/batched_transpose_c500_kernel.py`
- 原始 kernel 附件：`/Users/yiweihan/.codex/attachments/52987978-982f-4655-9714-5f7eb3851e3b/pasted-text.txt`
