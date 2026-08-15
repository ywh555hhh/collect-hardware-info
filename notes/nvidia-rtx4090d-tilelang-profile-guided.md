# TileLang 4090 Profile-Guided Adaptive Optimization Notes

Date: 2026-08-11

This note records the RTX 4090 D profiling-driven round for three TileLang kernels. The baseline is the DeepSeek `TileKernels` style implementation, which is already a strong expert-written GPU baseline. The goal on 4090 D was therefore modest: use `nsys`/`ncu` evidence where available, avoid correctness false positives, and identify any path-specific tuning that still survives full benchmark validation.

This is only one side of the larger arc. The clearer cross-hardware optimization story is: DeepSeek baseline -> modest RTX 4090 D tuning -> more substantial MetaX C500 adaptation. See `notes/tilelang-deepseek-baseline-4090-c500-arc.md`.

## Environment

- Remote GPU: NVIDIA GeForce RTX 4090 D
- CUDA/driver observed earlier in the same run: CUDA 12.8, driver 570.124.06
- Repository: `/data/src/TileKernels-Metax`
- Result root: `/data/results/tileops-4090`
- Profile-guided validation root: `/data/results/tileops-4090/profile_guided_adaptive_20260811`
- Local kernel snapshots:
  - `doc/per_channel_cast_fused_4090_profile_guided_kernel.py`
  - `doc/batched_transpose_4090_profile_guided_kernel.py`
  - `doc/swiglu_channel_cast_transpose_4090_profile_guided_kernel.py`

## Tooling Status

### NCU

`ncu` is installed, but hardware performance counters are blocked by the NVIDIA driver policy:

- `ncu` version: 2025.1.0
- Driver parameter: `RmProfilingAdminOnly: 1`
- Observed NCU failure: `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`

Conclusion: this machine currently cannot provide NCU stall/roofline/counter data unless the NVIDIA module is reloaded with unrestricted profiling counters. For this round, decisions were made from `nsys` single-kernel timings plus full benchmark/correctness sweeps.

### NSYS

`nsys` is usable and was run with CUDA profiler API capture ranges:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true
```

Representative kernel timings:

| Operator | Variant | Shape / path | NSYS kernel time |
|---|---:|---|---:|
| `batched_transpose` | baseline | bf16, hidden=7168, tokens=8064, experts=32 | 8,129,115 ns |
| `batched_transpose` | `block_k=8, threads=512` | same | 8,055,580 ns |
| `per_channel_cast_fused` | baseline plain | hidden=7168, tokens=4096, non-expand | 91,617 ns |
| `per_channel_cast_fused` | `plain TILE_K=256` | same | 90,016 ns |
| `per_channel_cast_fused` | baseline rescale | hidden=7168, tokens=4096, non-expand | 22,144 ns |
| `per_channel_cast_fused` | `rescale threads_per_token=16` | same | 22,112 ns |
| `swiglu_forward_and_per_channel_cast_and_transpose` | baseline | transpose, hidden=7168, tokens=8064, npt=32 | 318,050 ns |
| `swiglu_forward_and_per_channel_cast_and_transpose` | `T_TILE_X=128, T_TILE_Y=64` | same | 318,946 ns |

Important read: NSYS showed that some globally forced variants only produce tiny wins or even neutral/slightly negative timing on representative large shapes. That pushed the final implementation toward shape/path-specific defaults rather than global environment overrides.

Additional focused NSYS after the adaptive patch:

| Operator | Variant | Shape / path | NSYS kernel time |
|---|---:|---|---:|
| `per_channel_cast_fused` | old MoE plain default, `TILE_K=128` | hidden=7168, topk=6, experts=4, ep=64 | 418,819 ns |
| `per_channel_cast_fused` | adaptive MoE plain default, `TILE_K=256` | same | 381,635 ns |
| `swiglu_forward_and_per_channel_cast_and_transpose` | old transpose+npt32 tile, `32x256` | hidden=2048, tokens=4096, npt=32 | 13,632 ns |
| `swiglu_forward_and_per_channel_cast_and_transpose` | adaptive transpose+npt32 tile, `128x64` | same | 14,400 ns |

The focused PC MoE NSYS result aligns with the benchmark win and explains the main `per_channel` speedup. The focused SwiGLU single-kernel NSYS result is inconclusive because it conflicts with the 224-row benchmark suite, where transpose+npt32 improved `~2.04%` gmean and the full adaptive default improved `~0.51%` gmean. For SwiGLU, the accepted conclusion is therefore correctness-gated benchmark data, not a single small-shape NSYS capture.

Focused NSYS files:

- `/data/results/tileops-4090/profiling_20260811/nsys_adaptive_focus/pc_moe_plain_old/nsys_stats.txt`
- `/data/results/tileops-4090/profiling_20260811/nsys_adaptive_focus/pc_moe_plain_adaptive/nsys_stats.txt`
- `/data/results/tileops-4090/profiling_20260811/nsys_adaptive_focus/swiglu_t32_old/nsys_stats.txt`
- `/data/results/tileops-4090/profiling_20260811/nsys_adaptive_focus/swiglu_t32_adaptive/nsys_stats.txt`

## Shape-Diff Findings

### `per_channel_cast_fused`

Best prior global candidate:

- `TK_PC_TILE_K_PLAIN=256`
- `TK_PC_THREADS_PER_TOKEN_RESCALE=16`
- Repeat result: about `1.062x` gmean speedup, `~1.046 TB/s` gmean bandwidth, `~1.475 TB/s` peak

Shape-level diff showed the gain is concentrated in expanded MoE/top-k paths. Dense non-MoE cases were neutral to slightly negative when the plain `TILE_K=256` change was applied globally.

Final adaptive rule:

- If `with_expand` and plain input: default `TILE_K=256`
- If `with_expand` and rescale input: default `threads_per_token=16`
- Otherwise preserve the old dense defaults
- Environment knobs still override defaults

### `batched_transpose`

Raw earlier sweeps produced false positives:

- `block_k=8` and `threads=512` initially looked much faster, but failed correctness on remainder tiles.
- The accepted version fixed row coverage with ceil row loop bounds and `if i < row_tiles` guard.

Accepted default:

- `block_k=8`
- `num_threads=512`
- Keep 256-thread fallback for 64-wide tiles

This is not a large SOTA jump, but it is a real validated improvement:

- Full benchmark speedup: `~1.0054x`
- NSYS representative kernel time: `8.129 ms -> 8.056 ms`
- Best shape-level gain: `~1.019x` on e4m3, hidden=6144, experts=32, tokens=8064

### `swiglu_forward_and_per_channel_cast_and_transpose`

Global `128x64` was not acceptable as a default:

- Overall forced `128x64`: only `~1.0008x` gmean
- It improved transpose path with `num_per_tokens=32`
- It hurt several `without_transpose=True` paths

Path-specific diff:

- `without_transpose=False` and `num_per_tokens=32`: `~1.0204x` gmean speedup across 56 matched rows
- `without_transpose=True`: `~0.9917x`, so it must not inherit the same tile

Final adaptive rule:

- If transpose path and `num_per_tokens == 32`: use `TILE_X=128, TILE_Y=64`
- Otherwise preserve the existing heuristic
- Environment knobs still override defaults

## Final Validation Results

All three kernels were validated with default code behavior, with no optimization env overrides.

| Operator | Rows passed | Gmean BW | Max BW | Gmean speedup vs baseline | Top gain | Worst matched delta |
|---|---:|---:|---:|---:|---:|---:|
| `per_channel_cast_fused` | 96/96 | 1044.7 GB/s | 1474.8 GB/s | 1.0604x | 1.1156x | 0.9963x |
| `batched_transpose` | 84/84 | 904.7 GB/s | 932.7 GB/s | 1.0054x | 1.0190x | 0.9956x |
| `swiglu_forward_and_per_channel_cast_and_transpose` | 224/224 | 773.3 GB/s | 870.1 GB/s | 1.0051x | 1.0472x | 0.9945x |

Validation result files:

- `/data/results/tileops-4090/profile_guided_adaptive_20260811/per_channel/benchmark.jsonl`
- `/data/results/tileops-4090/profile_guided_adaptive_20260811/batched_transpose/benchmark.jsonl`
- `/data/results/tileops-4090/profile_guided_adaptive_20260811/swiglu/benchmark.jsonl`

## Resume-Ready Summary

Recommended conservative wording:

> Starting from DeepSeek TileKernels baselines, profiled and tuned three TileLang FP8/MoE data-movement kernels on RTX 4090 D using NSYS-guided profiling and correctness-gated benchmarking. Because the baseline was already strong, the 4090 D gains were modest: `1.060x` gmean for the MoE/top-k `per_channel_cast_fused` path and `~1.005x` whole-suite gains for `batched_transpose` and fused SwiGLU+FP8 cast+transpose.

Stronger wording is possible for the main win:

> Delivered a profile-guided adaptive TileLang FP8 quantization kernel on RTX 4090 D, improving the DeepSeek-baseline MoE/top-k `per_channel_cast_fused` path by `~6.0%` gmean and reaching `~1.475 TB/s` peak bandwidth while preserving dense-path behavior and passing all benchmark correctness cases.

Do not overclaim:

- `batched_transpose` is a small but real improvement, not a headline SOTA result.
- `swiglu` has meaningful path-specific improvement on transpose+npt32, but only modest whole-suite gain.
- NCU counter-level analysis is currently blocked by driver permissions, so final claims should say `NSYS-guided` rather than `NCU roofline-guided`.
- Do not imply the RTX 4090 D baseline was weak; it was the DeepSeek TileKernels baseline.
