# RTX 4090 D TileLang profile-guided kernel snapshots

These files are source snapshots from the final accepted profile-guided adaptive defaults:

- `per_channel_cast_fused_4090_profile_guided_kernel.py`
  - Dense path keeps upstream defaults.
  - Expanded/MoE plain path defaults to `TILE_K=256`.
  - Expanded/MoE rescale path defaults to `threads_per_token=16`.
- `batched_transpose_4090_profile_guided_kernel.py`
  - Defaults to `block_k=8` and `threads=512`.
  - Keeps a 256-thread fallback for 64-wide tiles.
- `swiglu_channel_cast_transpose_4090_profile_guided_kernel.py`
  - Uses `128x64` only for transpose path with `num_per_tokens=32`.
  - Keeps the original heuristic for `without_transpose=True` and other paths.

The corresponding experiment write-up is `notes/nvidia-rtx4090d-tilelang-profile-guided.md`.
