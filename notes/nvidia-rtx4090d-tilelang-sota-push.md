# TileLang RTX 4090 SOTA Push - 2026-08-11

Target: NVIDIA GeForce RTX 4090 D, CUDA 12.8, TileLang 0.1.13  
Remote result root: `/data/results/tileops-4090/sota_push_20260811/`  
Scope: continue pushing the three archived operators toward best-known 4090 results, with correctness checks before accepting any benchmark improvement.

## Final Accepted Results

| Operator | Best Accepted Variant | Baseline | Best Accepted | Speedup | Correctness |
|---|---|---:|---:|---:|---|
| `per_channel_cast_fused` | `TK_PC_TILE_K_PLAIN=256 TK_PC_THREADS_PER_TOKEN_RESCALE=16` | `985.2 GB/s` | `1046.0-1046.5 GB/s` | `~1.062x` | non-MoE `24/24`; benchmark `96/96` |
| `batched_transpose` | `TK_BT_BLOCK_K=8 TK_BT_NUM_THREADS=512` with fixed row coverage and safe 64-tile thread fallback | `899.9 GB/s` | `904.9 GB/s` | `~1.006x` | `84/84` |
| `swiglu_forward_and_per_channel_cast_and_transpose` | `TK_SWIGLU_T_TILE_X=128 TK_SWIGLU_T_TILE_Y=64` | `769.4 GB/s` | `773.3 GB/s` | `~1.005x` | `336/336` |

## `per_channel_cast_fused`

Already-best combined setting:

| Variant | Rows | GMean BW | Max BW | Plain BW | Rescale BW | Status |
|---|---:|---:|---:|---:|---:|---|
| baseline warp-32 | 96 | `985.2 GB/s` | `1.345 TB/s` | `1122.2 GB/s` | `865.0 GB/s` | baseline |
| `plain256 + rescale16` | 96 | `1046.0-1046.5 GB/s` | `1.470-1.475 TB/s` | `1188.2-1188.4 GB/s` | `920.8-921.5 GB/s` | accepted best |
| `plain512 + rescale16` | 48 usable rows | n/a | n/a | n/a | n/a | rejected: plain path failures |
| `plain256 + rescale8` | 96 | `1023.8 GB/s` | `1.473 TB/s` | `1186.2 GB/s` | `883.6 GB/s` | worse than rescale16 |
| `plain256 + rescale4` | 96 | `987.8 GB/s` | `1.472 TB/s` | `1187.4 GB/s` | `821.7 GB/s` | rejected |

Conclusion: `threads_per_token=16` is the best observed rescale sweet spot; going smaller hurts rescale bandwidth. `TILE_K_PLAIN=512` is not safe across benchmark shapes.

## `batched_transpose`

The high-payoff path was making `block_k=8` correct on 64-wide remainder tiles. The bug was the old floor-divided row loop:

```python
block_x // block_k // (num_threads // num_threads_per_row)
```

For `block_k=8` and 64-wide remainder tiles this can become zero, leaving rows unread. The accepted fix uses ceil coverage plus an `i < row_tiles` guard.

| Variant | Rows | GMean BW | Max BW | Correctness | Status |
|---|---:|---:|---:|---|---|
| baseline | 84 | `899.9 GB/s` | `926.4 GB/s` | pass | baseline |
| raw `block_k=8` | 84 | `1329.6 GB/s` | `1946.5 GB/s` | fail | false positive |
| safe fallback `block_k=8` | 84 | `899.8 GB/s` | `925.7 GB/s` | `84/84` | no gain |
| fixed coverage `block_k=8` | 84 | `902.2 GB/s` | `933.1 GB/s` | `84/84` | accepted intermediate |
| fixed coverage `block_k=8 + threads512` | 84 | `904.9 GB/s` | `932.7 GB/s` | `84/84` | accepted best |

Conclusion: a real SOTA-level jump was not achieved yet, but the earlier false high-bandwidth path was converted into a correct implementation and yielded a small gain. Next step is redesigning the writeback/layout path; bigger read tiles alone do not preserve the `1.3 TB/s` false-positive number once all rows are covered.

## `swiglu_forward_and_per_channel_cast_and_transpose`

Best accepted variant remains the transpose-only tile override:

| Variant | Rows | GMean BW | Max BW | Correctness | Status |
|---|---:|---:|---:|---|---|
| baseline | 224 | `769.4 GB/s` | `870.4 GB/s` | pass | baseline |
| `T_TILE_X=128 T_TILE_Y=64` | 224 | `773.3 GB/s` | `870.1 GB/s` | `336/336` | accepted best |
| `T_TILE_X=64 T_TILE_Y=64` | 168 partial | `773.3 GB/s` partial | `871.1 GB/s` | fail rows | rejected |
| `T_TILE_X=64 T_TILE_Y=128` | 160 partial | `779.5 GB/s` partial | `872.0 GB/s` | fail rows | rejected |
| `T_TILE_X=128 T_TILE_Y=128` | 208 partial | `769.7 GB/s` partial | `870.2 GB/s` | fail rows | rejected |
| raw `TILE_K=8` | 121 partial | high partial BW | high | byte-level fail | rejected false positive |

Best path split:

| Split | Baseline | Best Accepted | Speedup |
|---|---:|---:|---:|
| transpose, `num_per_tokens=32`, no clamp | `753.8 GB/s` | `769.3 GB/s` | `~1.021x` |
| transpose, `num_per_tokens=32`, clamp `0.5` | `754.2 GB/s` | `769.6 GB/s` | `~1.020x` |

Conclusion: the correct gain is small but path-specific and stable. More aggressive transpose tile shapes produce partial benchmark gains but fail some benchmark shapes, so they are not acceptable SOTA claims.

## Resume Framing

- Achieved a strong RTX 4090/Ada tuning result for a TileLang FP8 MoE quantization kernel: `~985 GB/s -> ~1.046 TB/s` gmean and `~1.47 TB/s` peak effective bandwidth.
- Extended the optimization effort across three TileLang kernels with correctness-gated ablation, separating real wins from benchmark-only false positives.
- Fixed a `batched_transpose` remainder-tile coverage issue exposed by larger register tiles; converted a failing `block_k=8` variant into a correct implementation and improved gmean bandwidth to `~905 GB/s`.
- Added path-specific SwiGLU transpose tuning with full correctness coverage, improving transpose/`num_per_tokens=32` bandwidth by about `2%`.

## Next High-Upside Work

1. `batched_transpose`: redesign the shared-memory writeback/readback mapping after the coverage fix; this is where the lost false-positive bandwidth likely went.
2. `swiglu`: create a conditional implementation that applies more aggressive tile shapes only to passing shape families, or derives a separate small-hidden fallback.
3. `per_channel_cast_fused`: add deterministic MoE correctness and profile the accepted combined variant with NCU.
