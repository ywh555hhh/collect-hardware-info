#!/usr/bin/env bash
set -uo pipefail
cd /data/src/TileKernels-Metax || exit 1
ROOT=/data/results/tileops-4090/profiling_20260811/nsys
mkdir -p "$ROOT"
run_one() {
  name="$1"; shift
  out="$ROOT/$name"; mkdir -p "$out"
  echo "=== NSYS $name ==="
  env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$out/report" \
    /data/venvs/ai-infra/bin/python /tmp/profile_tileops_kernel.py > "$out/nsys_profile.txt" 2>&1
  echo $? > "$out/status.txt"
  tail -18 "$out/nsys_profile.txt" || true
  if [ -f "$out/report.nsys-rep" ]; then
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum "$out/report.nsys-rep" > "$out/nsys_stats.txt" 2>&1 || true
    sed -n '1,150p' "$out/nsys_stats.txt"
  fi
}
run_one pc_plain_baseline env PROFILE_OP=pc_plain HIDDEN=7168 NUM_TOKENS=4096
run_one pc_plain_tile256 env PROFILE_OP=pc_plain HIDDEN=7168 NUM_TOKENS=4096 TK_PC_TILE_K_PLAIN=256
run_one pc_rescale_baseline env PROFILE_OP=pc_rescale HIDDEN=7168 NUM_TOKENS=4096
run_one pc_rescale_tpt16 env PROFILE_OP=pc_rescale HIDDEN=7168 NUM_TOKENS=4096 TK_PC_THREADS_PER_TOKEN_RESCALE=16
run_one swiglu_t_baseline env PROFILE_OP=swiglu_t HIDDEN=7168 NUM_TOKENS=8064 NPT=32 CLAMP=none
run_one swiglu_t_128x64 env PROFILE_OP=swiglu_t HIDDEN=7168 NUM_TOKENS=8064 NPT=32 CLAMP=none TK_SWIGLU_T_TILE_X=128 TK_SWIGLU_T_TILE_Y=64
