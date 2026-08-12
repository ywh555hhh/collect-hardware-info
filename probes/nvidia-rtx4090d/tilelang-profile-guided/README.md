# RTX 4090 D TileLang profiling and benchmark probes

These scripts were used to reproduce the TileLang/TileKernels-Metax 4090 D experiments.

## Scripts

- `run_profile_guided_adaptive.sh`: final no-env-override validation for all three adaptive defaults.
- `profile_tileops_kernel.py`: small NSYS capture harness for representative dense/transpose paths.
- `profile_tileops_adaptive_focus.py`: focused NSYS harness for PC MoE and SwiGLU transpose+npt32 paths.
- `run_nsys_profiles_more.sh`: NSYS batch runner for baseline vs selected candidates.
- `run_pc_selected_repeat.sh`: repeat selected `per_channel_cast_fused` candidates.
- `run_bt_swiglu_sweep.sh`: earlier `batched_transpose` and SwiGLU sweep runner.
- `run_sota_more.sh`: later sweep runner for additional rejected/accepted variants.
- `summarize_pc_sweep.py`: helper for summarizing PC sweep JSONL outputs.

## Expected remote layout

The scripts assume the original experiment layout:

```text
/data/src/TileKernels-Metax
/data/venvs/ai-infra/bin/python
/data/results/tileops-4090
```

Adjust those paths before reuse on another machine.
