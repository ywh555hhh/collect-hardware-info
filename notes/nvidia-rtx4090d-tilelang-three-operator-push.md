# TileLang RTX 4090 Three-Operator Push - 2026-08-11

Target: NVIDIA GeForce RTX 4090 D, CUDA 12.8, TileLang 0.1.13  
Remote result root: `/data/results/tileops-4090/three_ops_push_20260811/`  
Related prior doc: `doc/tilelang_4090_adaptive_per_channel_sweep_20260811.md`

## Summary

This pass pushed all three archived C500 migration operators on RTX 4090/Ada instead of only optimizing `per_channel_cast_fused`.

| Operator | Best Correct 4090 Result | Status |
|---|---:|---|
| `per_channel_cast_fused` | `~1.046 TB/s` gmean, `~1.47 TB/s` peak, `~1.06x` over repeat baseline | Strong positive result |
| `batched_transpose` | no correct aggregate speedup yet; best safe variants return to `~900 GB/s` baseline | Negative result after rejecting false positives |
| `swiglu_forward_and_per_channel_cast_and_transpose` | `773.3 GB/s` vs `769.4 GB/s` gmean, `~1.005x`; transpose+npt32 split improves `~754 -> ~769 GB/s` | Small but correct path-specific positive |

## 1. `per_channel_cast_fused`

The best result remains the combined-path Ada tuning from the previous deep pass:

| Variant | Rows | GMean BW | Max BW | Speed vs Repeat Baseline | Plain BW | Rescale BW |
|---|---:|---:|---:|---:|---:|---:|
| baseline warp-32 | 96 | `985.2 GB/s` | `1.345 TB/s` | `1.000x` | `1122.2 GB/s` | `865.0 GB/s` |
| `TK_PC_TILE_K_PLAIN=256 TK_PC_THREADS_PER_TOKEN_RESCALE=16` | 96 | `1046.0-1046.5 GB/s` | `1.470-1.475 TB/s` | `~1.062x` | `1188.2-1188.4 GB/s` | `920.8-921.5 GB/s` |

Validation:

- Benchmark rows: `96/96` passed for selected variants.
- Non-MoE reference correctness: `24/24` byte-level pass for the best combined setting.
- MoE/topk deterministic correctness still needs an in-process fixed-input fixture.

Useful resume claim:

> Tuned a TileLang FP8 MoE per-channel quantization kernel on RTX 4090 by combining plain-path `TILE_K=256` with rescale-path `threads_per_token=16`, improving gmean bandwidth from `~985 GB/s` to `~1.046 TB/s` and reaching `~1.47 TB/s` peak effective bandwidth.

## 2. `batched_transpose`

Baseline was remeasured before new 4090 experiments:

| Variant | Rows | GMean BW | Max BW | Correctness |
|---|---:|---:|---:|---|
| baseline | 84 | `899.9 GB/s` | `926.4 GB/s` | existing test passes |
| `TK_BT_BLOCK_K=2` | 84 | `890.4 GB/s` | `924.9 GB/s` | not useful |
| raw `TK_BT_BLOCK_K=8` | 84 | `1329.6 GB/s` | `1946.5 GB/s` | rejected: correctness fails on 64 remainder tiles |
| raw `TK_BT_NUM_THREADS=512` | 84 | `950.5 GB/s` | `1951.4 GB/s` | rejected: correctness fails on 64 remainder tiles |
| safe `TK_BT_BLOCK_K=8` with 64-tile fallback | 84 | `899.8 GB/s` | `925.7 GB/s` | `84/84` pass |
| safe `TK_BT_NUM_THREADS=512` with 64-tile fallback | 84 | `899.9 GB/s` | `925.4 GB/s` | `84/84` pass |

Interpretation:

- The apparent `block_k=8` and `512-thread` wins were false positives: benchmark-only timing did not check output correctness.
- Both failed specifically around remainder kernels where either `block_x` or `block_y` becomes `64`; the loop coverage formula can leave rows unread.
- Adding correctness-safe fallbacks removes the speedup, which means the next real optimization needs a new row-coverage/layout formula for 64 remainder tiles, not only bigger register tiles or more threads.

Useful resume framing:

> Built a correctness-gated RTX 4090 ablation for a TileLang batched transpose kernel and rejected benchmark-only false positives where `block_k=8`/512-thread variants reached `1.3-1.9 TB/s` but failed remainder-tile correctness; isolated the real blocker to 64-wide remainder-tile row coverage.

Local snapshot: `doc/batched_transpose_4090_adaptive_kernel.py`.

## 3. `swiglu_forward_and_per_channel_cast_and_transpose`

Baseline and path-specific sweep:

| Variant | Rows | GMean BW | Max BW | Correctness |
|---|---:|---:|---:|---|
| baseline | 224 | `769.4 GB/s` | `870.4 GB/s` | existing test passes |
| `TK_SWIGLU_TILE_K=2` | 93 partial before timeout | `696.3 GB/s` partial | `823.6 GB/s` | not useful |
| `TK_SWIGLU_TILE_K=8` | 121 partial before timeout | high partial BW | rejected: byte-level correctness fails |
| global `TK_SWIGLU_TILE_X=128 TK_SWIGLU_TILE_Y=64` | 224 | `770.1 GB/s` | `854.0 GB/s` | benchmark pass |
| `TK_SWIGLU_T_TILE_X=128 TK_SWIGLU_T_TILE_Y=64` | 224 | `773.3 GB/s` | `870.1 GB/s` | `336/336` full correctness pass |
| `TK_SWIGLU_NT_TILE_X=128 TK_SWIGLU_NT_TILE_Y=64` | 224 | `766.2 GB/s` | `854.2 GB/s` | not useful |

Path split for the useful `t_128x64` variant:

| Path Split | Baseline GMean BW | `t_128x64` GMean BW | Result |
|---|---:|---:|---|
| transpose, `num_per_tokens=32`, no clamp | `753.8 GB/s` | `769.3 GB/s` | `~1.021x` |
| transpose, `num_per_tokens=32`, clamp `0.5` | `754.2 GB/s` | `769.6 GB/s` | `~1.020x` |
| transpose, `num_per_tokens=128` | `~763 GB/s` | `~763 GB/s` | neutral |
| no-transpose, `num_per_tokens=32` | `~790 GB/s` | `~790 GB/s` | preserved |
| no-transpose, `num_per_tokens=128` | `~771 GB/s` | `~771 GB/s` | preserved |

Interpretation:

- A path-specific transpose tile override gives a small but real improvement without hurting no-transpose paths.
- Raw `TILE_K=8` is not usable despite high partial benchmark numbers; it fails byte-level correctness against the PyTorch reference.
- The next high-upside direction is not global tile changes, but a dedicated transpose/npt32 implementation or conditional tile policy.

Useful resume framing:

> Added path-specific Ada tuning for a fused SwiGLU + per-channel FP8 cast + transpose TileLang kernel; improved the transpose/`num_per_tokens=32` path by about `2%` while preserving no-transpose performance, with `336/336` correctness coverage.

Local snapshot: `doc/swiglu_channel_cast_transpose_4090_adaptive_kernel.py`.

## Recommended Next Push

1. `batched_transpose`: rewrite the row coverage formula so `block_k=8` works for `64` remainder tiles; this is the only path that showed a large latent speedup.
2. `swiglu`: add conditional tile policy for transpose + `num_per_tokens=32`, then test whether the `128x64` override can be narrowed to only the profitable path.
3. `per_channel_cast_fused`: add deterministic MoE/topk correctness fixture and run NCU for the combined winner.
