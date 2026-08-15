# TileLang DeepSeek baseline -> RTX 4090 D -> MetaX C500 arc

Date: 2026-08-15

This note clarifies the intended story for the TileLang operator work. The baseline is the DeepSeek `TileKernels` style implementation, not an unoptimized toy kernel.

## Correct Narrative

1. Start from the DeepSeek `TileKernels` operator baseline.
2. Test and tune that baseline on RTX 4090 D with NSYS and correctness-gated benchmark data.
3. Observe that 4090 D improvements are real but modest because the DeepSeek baseline is already strong.
4. Port and optimize the same operator family for MetaX C500, where hardware differences and C500-specific dataflow changes make the optimization story more visible.

## Why the 4090 D Result Is Modest

The DeepSeek baseline already contains expert-level GPU kernel decisions: fused data movement, shared-memory staging, vectorized loads/stores, and shape-aware TileLang structure. On RTX 4090 D, most local changes are therefore marginal unless they change a real path-specific bottleneck.

The most meaningful 4090 D improvement was:

- `per_channel_cast_fused`: `1.0604x` gmean speedup, `1474.8 GB/s` peak bandwidth, mostly from MoE/top-k expand path tuning.

The other two whole-suite gains are correctly described as small:

- `batched_transpose`: `1.0054x`
- `swiglu_forward_and_per_channel_cast_and_transpose`: `1.0051x`

## Why C500 Is the Stronger Optimization Story

The C500 work is more than parameter tuning. It involves adapting TileLang operator structure to a different accelerator execution model and memory behavior:

- C500 warp/wavefront and thread-mapping assumptions differ from NVIDIA.
- Shared-memory pressure, bank behavior, vectorization choices, and tile sizing differ.
- Several C500 optimizations required dataflow changes rather than simple environment knobs.
- The C500 artifacts therefore tell a stronger cross-hardware kernel-porting story.

Relevant C500 operator notes:

- `notes/metax-c500-tilelang/batched_transpose_c500_optimization.md`
- `notes/metax-c500-tilelang/quant_per_channel_cast_fused_c500_optimization.md`
- `notes/metax-c500-tilelang/quant_swiglu_channel_cast_transpose_c500_optimization.md`

## Resume-Safe Wording

Recommended wording:

> Starting from DeepSeek TileKernels baselines, profiled and tuned three TileLang FP8/MoE data-movement operators on RTX 4090 D, then ported and reworked the same operator family for MetaX C500. The 4090 D tuning produced a modest but correctness-gated `1.060x` gmean win on the MoE/top-k `per_channel_cast_fused` path, while the C500 work required more substantial hardware-specific dataflow and memory-layout changes.

Avoid:

- Do not imply the 4090 D baseline was weak.
- Do not claim large 4090 D SOTA gains across all three operators.
- Do not present C500 ideas as simply transferring unchanged to NVIDIA.
- Do not claim NCU counter-level analysis for the 4090 D run; NCU was blocked by driver policy.
