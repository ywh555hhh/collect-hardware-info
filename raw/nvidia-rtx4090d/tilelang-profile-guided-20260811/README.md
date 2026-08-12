# RTX 4090 D TileLang profile-guided raw results

采集时间：2026-08-11

这组 raw 数据来自 TileKernels-Metax 在 RTX 4090 D 上的三算子 profile-guided 优化实验。这里只保留文本和 JSONL 证据，未保存 `.nsys-rep` / sqlite 等大二进制 profile 文件。

## Included data

- `profile_guided_adaptive_20260811/*/benchmark.jsonl`: 三个算子最终 adaptive 默认策略的完整 benchmark 记录。
- `profile_guided_adaptive_20260811/*/pytest.log`: 对应 pytest benchmark 输出，包含通过行数。
- `profiling_20260811/nsys/*/nsys_stats.txt`: 代表 shape 的 NSYS kernel/API summary。
- `profiling_20260811/nsys_adaptive_focus/*/nsys_stats.txt`: adaptive patch 后针对 PC MoE 和 SwiGLU transpose+npt32 的 focused NSYS。
- `opt_sweep/.../benchmark.jsonl`, `three_ops_push.../benchmark.jsonl`, `sota_push.../benchmark.jsonl`: baseline 和关键候选对照数据。

## Key final metrics

| Operator | Rows | Gmean BW | Peak BW | Gmean speedup |
| --- | ---: | ---: | ---: | ---: |
| `per_channel_cast_fused` | 96 | 1044.7 GB/s | 1474.8 GB/s | 1.0604x |
| `batched_transpose` | 84 | 904.7 GB/s | 932.7 GB/s | 1.0054x |
| `swiglu_forward_and_per_channel_cast_and_transpose` | 224 | 773.3 GB/s | 870.1 GB/s | 1.0051x |

## Notes

- NCU was installed but counter collection was blocked by `RmProfilingAdminOnly=1`, so the reliable profiler evidence in this snapshot is NSYS plus correctness-gated benchmark JSONL.
- Pytest benchmark exits may be non-zero because the benchmark plugin has no stored baseline; use the JSONL row count and pytest log pass count as the correctness signal.
