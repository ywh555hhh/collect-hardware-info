# MetaX C500 TileLang operator optimization notes

These notes preserve the C500 side of the TileLang operator work. They should be read together with the RTX 4090 D profile-guided results:

- `notes/tilelang-deepseek-baseline-4090-c500-arc.md`
- `notes/nvidia-rtx4090d-tilelang-profile-guided.md`

The intended story is:

1. DeepSeek `TileKernels` provides a strong baseline.
2. RTX 4090 D profile-guided tuning produced modest but real gains.
3. MetaX C500 required more visible hardware-specific operator adaptation.

## Notes

- `batched_transpose_c500_optimization.md`
- `quant_per_channel_cast_fused_c500_optimization.md`
- `quant_swiglu_channel_cast_transpose_c500_optimization.md`

## Kernel Snapshots

The corresponding C500 kernel snapshots are under:

```text
kernels/metax-c500/tilelang-ops/
```
