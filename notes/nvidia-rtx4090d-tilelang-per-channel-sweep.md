# TileLang RTX 4090 Ada Optimization Sweep - `per_channel_cast_fused`

Date: 2026-08-11  
Target: NVIDIA GeForce RTX 4090 D, SM 8.9, CUDA 12.8  
Repos: `TileKernels-Metax` commit `0266ab740980de7dc03a828b8259cd73d100c2eb`, `TileKernels-deepseek` commit `36d9e45d38e204ebb87e6f6e833821eee0482fe5`  
Remote results root: `/data/results/tileops-4090/opt_sweep/`

## Goal

After the first C500-to-4090 ablation showed that most C500-specific tile/register heuristics do not transfer, this sweep tried a more NVIDIA/Ada-specific tuning direction for `quant_per_channel_cast_fused`:

- keep the CUDA warp-32 semantic fix from the C500 porting pass;
- avoid C500-style register staging, which previously regressed 4090 performance;
- sweep the two actual hot path controls independently:
  - plain output path: hidden-dimension tile size `TILE_K`;
  - fused cast-back/rescale path: `num_threads_per_token`.

## Kernel Change

The source was changed to expose compile-time environment knobs while preserving upstream/MetaX default behavior when the knobs are unset:

| Knob | Default | Meaning |
|---|---:|---|
| `TK_PC_TILE_K_PLAIN` | `128` | plain path hidden tile width |
| `TK_PC_THREADS_PER_TOKEN_PLAIN` | `64` | plain path thread grouping |
| `TK_PC_TILE_K_RESCALE` | `256` | rescale/cast-back path hidden tile width |
| `TK_PC_THREADS_PER_TOKEN_RESCALE` | `64` | rescale/cast-back path thread grouping |
| `TK_PC_REGISTER_STAGING` | `0` | optional plain-path local/register staging ablation |

Local kernel snapshot: `doc/per_channel_cast_fused_4090_adaptive_kernel.py`.

## Sweep Results

Primary sweep: `/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811/SUMMARY.md`

| Variant | Rows | GMean Latency (us) | GMean BW (GB/s) | Max BW (GB/s) | Speed vs Baseline | Plain BW | Rescale BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rescale_tpt16` | 96 | 154.70 | 1013.9 | 1334.2 | 1.032x | 1118.0 | 919.4 |
| `rescale_tpt32` | 96 | 154.79 | 1013.3 | 1343.9 | 1.031x | 1117.5 | 918.8 |
| `plain_tile256` | 96 | 155.01 | 1011.9 | 1468.0 | 1.030x | 1186.3 | 863.1 |
| `baseline_warp32` | 96 | 159.64 | 982.5 | 1335.3 | 1.000x | 1118.4 | 863.2 |
| `rescale_tile128_tpt32` | 96 | 161.78 | 969.5 | 1333.6 | 0.987x | 1116.7 | 841.7 |
| `plain_tpt32` | 96 | 169.91 | 923.1 | 1116.4 | 0.940x | 989.8 | 860.9 |
| `plain_tile256_tpt32` | 96 | 170.24 | 921.3 | 1156.8 | 0.938x | 984.1 | 862.6 |
| `rescale_tile128` | 96 | 186.85 | 839.4 | 1334.3 | 0.854x | 1116.9 | 630.9 |

Selected repeat run: `/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811_repeat1/SUMMARY.md`

| Variant | Rows | GMean Latency (us) | GMean BW (GB/s) | Max BW (GB/s) | Speed vs Baseline | Plain BW | Rescale BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rescale_tpt16` | 96 | 154.33 | 1016.3 | 1344.7 | 1.032x | 1121.9 | 920.6 |
| `rescale_tpt32` | 96 | 154.41 | 1015.8 | 1345.7 | 1.031x | 1122.4 | 919.3 |
| `plain_tile256` | 96 | 154.74 | 1013.6 | 1473.6 | 1.029x | 1187.8 | 865.0 |
| `baseline_warp32` | 96 | 159.20 | 985.2 | 1345.0 | 1.000x | 1122.2 | 865.0 |

Combined-path sweep: `/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811_combined/`

| Variant | Rows | GMean Latency (us) | GMean BW (GB/s) | Max BW (GB/s) | Speed vs Repeat Baseline | Plain BW | Rescale BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `combined_tile256_rescale_tpt16` | 96 | 149.95 | 1046.0 | 1474.8 | 1.062x | 1188.2 | 920.8 |
| `combined_tile256_rescale_tpt32` | 96 | 150.05 | 1045.3 | 1473.2 | 1.061x | 1188.4 | 919.5 |

Second confirmation run for best combined setting: `/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811_combined_repeat2/combined_tile256_rescale_tpt16/`

| Variant | Rows | GMean Latency (us) | GMean BW (GB/s) | Max BW (GB/s) | Speed vs Repeat Baseline | Plain BW | Rescale BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `combined_tile256_rescale_tpt16` | 96 | 149.88 | 1046.5 | 1470.1 | 1.062x | 1188.4 | 921.5 |

## Interpretation

- Best combined setting: `TK_PC_TILE_K_PLAIN=256` plus `TK_PC_THREADS_PER_TOKEN_RESCALE=16`. This combines the independent plain-path and rescale-path wins, raising repeat gmean bandwidth from `985.2 GB/s` to `1046.0-1046.5 GB/s`, about `1.062x`.
- Best aggregate setting: `TK_PC_THREADS_PER_TOKEN_RESCALE=16` or `32`. Both are stable across the primary and repeat runs, improving mixed-path gmean bandwidth by about `3.1-3.2%` over the warp-32 baseline.
- Best plain-path setting: `TK_PC_TILE_K_PLAIN=256`. This raises plain-path gmean bandwidth from about `1122 GB/s` to `1188 GB/s` on the repeat run and increases observed peak effective bandwidth to `1.474 TB/s`.
- Important path split: `plain_tile256` improves plain output but leaves rescale unchanged; `rescale_tpt16/tpt32` improves rescale/cast-back but leaves plain unchanged. These should be treated as workload-specific knobs, not one universal replacement.
- Negative result: reducing plain-path `threads_per_token` to `32` regresses badly. This reinforces that the C500 habit of reducing per-token parallelism does not directly carry to Ada.
- Negative result: `rescale_tile128` preserves plain-path bandwidth but damages rescale bandwidth, so smaller rescale tiles are not useful on 4090.

## Validation Notes

- The benchmark suite produced `96/96` passing benchmark rows for every selected variant. The pytest process exits with status `1` because no benchmark baselines are registered; this is a benchmark-plugin artifact, not row failure.
- A separate byte-level correctness check against the PyTorch reference passed for all non-MoE benchmark shapes: `24/24` rows for baseline, `plain_tile256`, `rescale_tpt16`, and the best combined setting `TK_PC_TILE_K_PLAIN=256 TK_PC_THREADS_PER_TOKEN_RESCALE=16`.
- MoE/topk rows are harder to compare cross-process because the generated random top-k input differs between separate runs. The v2 fingerprint run confirmed `72/96` input mismatches across processes exactly on the topk rows, so cross-process output-hash mismatch is not used as a correctness claim.
- The current performance claim is therefore strongest for benchmark throughput and non-MoE numerical correctness. For a publishable upstream PR, add an in-process deterministic MoE fixture that reuses identical `x`, `x_sf`, and `pos_to_token` across variants.

## Resume-Ready Framing

- Built a reproducible RTX 4090/Ada TileLang ablation harness for DeepSeek/MetaX FP8 MoE quantization kernels, covering 96 benchmark shapes across plain and fused cast-back paths.
- Identified architecture-specific tuning for `per_channel_cast_fused`: combining plain-path `TILE_K=256` with rescale-path `threads_per_token=16` improved aggregate gmean bandwidth from `~985 GB/s` to `~1.046 TB/s` on repeated measurement (`~1.06x`).
- Tuned the plain FP8 quantization path by increasing `TILE_K` from 128 to 256, raising plain-path gmean bandwidth to `~1.188 TB/s` and peak effective bandwidth to `~1.474 TB/s` on RTX 4090 D.
- Demonstrated that several C500-derived heuristics do not transfer to Ada, using path-split benchmarks and repeat measurements instead of one-off speedup claims.

## Next Experiments

1. Add a deterministic MoE/topk correctness fixture that fixes `topk_idx`, `pos_to_token`, and expanded input tensors in one process.
2. Use NCU on the best combined setting to confirm whether the improvement comes from memory transaction efficiency, occupancy, or reduced shared-memory pressure.
3. Try a narrow plain-path tile sweep around the winner, especially `TILE_K=192/320/384` if TileLang accepts the resulting vectorization and shared-memory shape.
4. Move to `quant_swiglu_channel_cast_transpose`: profile first, then sweep path-specific thread mapping and transpose/no-transpose layout separately.
