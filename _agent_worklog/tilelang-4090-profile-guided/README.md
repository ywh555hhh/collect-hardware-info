# TileLang RTX 4090 D profile-guided kernel optimization worklog

Date: 2026-08-15

This worklog archives the TileLang / TileKernels-Metax operator optimization session. The correct arc is: start from the DeepSeek `TileKernels` baseline, tune that strong baseline on RTX 4090 D with NSYS-guided profiling and correctness-gated benchmark data, then preserve the C500-side operator optimization notes where the hardware-specific migration/rework is more visible.

## Scope

Three kernels were investigated:

- `per_channel_cast_fused`
- `batched_transpose`
- `swiglu_forward_and_per_channel_cast_and_transpose`

The durable output is not a claim that all three reached a new SOTA. The durable output is a reproducible profile-guided optimization record with explicit claim boundaries and a clear baseline story.

## Achieved Work

- Preserved DeepSeek-baseline -> RTX 4090 D -> MetaX C500 narrative notes and RTX 4090 D benchmark/profiling records.
- Preserved C500 TileLang operator optimization notes and final C500 kernel snapshots.
- Added final profile-guided kernel snapshots under `kernels/nvidia-rtx4090d/tilelang-profile-guided/`.
- Preserved benchmark JSONL, pytest logs, and NSYS stats under `raw/nvidia-rtx4090d/tilelang-profile-guided-20260811/`.
- Preserved reusable benchmark/profile scripts under `probes/nvidia-rtx4090d/tilelang-profile-guided/`.
- Updated the repository root `README.md` with final metrics and claim boundaries.

## Final Metrics

| Operator | Rows | Gmean BW | Peak BW | Gmean speedup |
| --- | ---: | ---: | ---: | ---: |
| `per_channel_cast_fused` | 96/96 | 1044.7 GB/s | 1474.8 GB/s | 1.0604x |
| `batched_transpose` | 84/84 | 904.7 GB/s | 932.7 GB/s | 1.0054x |
| `swiglu_forward_and_per_channel_cast_and_transpose` | 224/224 | 773.3 GB/s | 870.1 GB/s | 1.0051x |

## Claim Boundary

Safe external claim:

- Profile-guided adaptive TileLang FP8/MoE kernel optimization on RTX 4090 D.
- `per_channel_cast_fused` MoE/top-k path delivered the strongest result: about `1.060x` gmean speedup and about `1.475 TB/s` peak bandwidth.
- `batched_transpose` and SwiGLU whole-suite wins are real but small; they should be described as correctness-gated incremental improvements, not large SOTA breakthroughs.
- The baseline should be named as the DeepSeek TileKernels baseline.
- The more visible hardware-specific optimization story is the C500 adaptation/rework, not the modest 4090 D local tuning.

Non-claims:

- Do not claim production readiness.
- Do not claim NCU roofline/stall analysis; NCU counters were blocked by driver policy.
- Do not claim all three kernels achieved major SOTA improvements.
- Do not claim strict cross-hardware portability of C500-specific tuning rules.
- Do not imply the 4090 D baseline was weak or naive.

## Key Paths

- Main note: `notes/nvidia-rtx4090d-tilelang-profile-guided.md`
- Baseline arc note: `notes/tilelang-deepseek-baseline-4090-c500-arc.md`
- C500 TileLang notes: `notes/metax-c500-tilelang/`
- Per-channel sweep note: `notes/nvidia-rtx4090d-tilelang-per-channel-sweep.md`
- Three-operator push note: `notes/nvidia-rtx4090d-tilelang-three-operator-push.md`
- SOTA push note: `notes/nvidia-rtx4090d-tilelang-sota-push.md`
- Kernel snapshots: `kernels/nvidia-rtx4090d/tilelang-profile-guided/`
- C500 kernel snapshots: `kernels/metax-c500/tilelang-ops/`
- Reproduction scripts: `probes/nvidia-rtx4090d/tilelang-profile-guided/`
- Raw benchmark/profiling data: `raw/nvidia-rtx4090d/tilelang-profile-guided-20260811/`

## Reproduction Notes

The scripts assume the original remote layout:

```text
/data/src/TileKernels-Metax
/data/venvs/ai-infra/bin/python
/data/results/tileops-4090
```

The most important validation entry point is:

```bash
probes/nvidia-rtx4090d/tilelang-profile-guided/run_profile_guided_adaptive.sh
```

## Archive Reference

- GitHub repository: `https://github.com/ywh555hhh/collect-hardware-info`
- Branch: `tilelang-4090-profile-guided-results`
- Commit before this worklog: `fd2076b8a3f091ddda064e1c16feb571773cbf86`
- Pull request URL: `https://github.com/ywh555hhh/collect-hardware-info/pull/new/tilelang-4090-profile-guided-results`
