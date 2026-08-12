#!/usr/bin/env bash
set -uo pipefail
cd /data/src/TileKernels-Metax || exit 1
ROOT=/data/results/tileops-4090/sota_push_20260811
mkdir -p "$ROOT"

summ_pc() {
  name="$1"; out="$ROOT/per_channel/$name"
  /data/venvs/ai-infra/bin/python - <<PY
import json, pathlib, statistics
out=pathlib.Path('$out'); name='$name'
rows=[json.loads(x) for x in (out/'benchmark.jsonl').read_text().splitlines() if x.strip()] if (out/'benchmark.jsonl').exists() else []
print('PC_SUMMARY', name, 'rows', len(rows))
if rows:
 b=[r['bandwidth_gbs'] for r in rows]; t=[r['time_us'] for r in rows]
 plain=[r['bandwidth_gbs'] for r in rows if not r['params']['is_fused_cast_back']]
 rescale=[r['bandwidth_gbs'] for r in rows if r['params']['is_fused_cast_back']]
 print('gmean_us', statistics.geometric_mean(t), 'gmean_bw', statistics.geometric_mean(b), 'max_bw', max(b), 'plain_bw', statistics.geometric_mean(plain), 'rescale_bw', statistics.geometric_mean(rescale))
PY
}
run_pc() {
  name="$1"; shift
  out="$ROOT/per_channel/$name"; mkdir -p "$out"
  echo "=== PC_MORE $name ==="
  timeout 240s env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    /data/venvs/ai-infra/bin/python -m pytest -q tests/quant/test_per_channel_cast_fused.py::test_per_channel_cast_fused_benchmark \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose > "$out/pytest.log" 2>&1
  echo $? > "$out/status.txt"; tail -12 "$out/pytest.log" || true; summ_pc "$name"
}

summ_swiglu() {
  name="$1"; out="$ROOT/swiglu/$name"
  /data/venvs/ai-infra/bin/python - <<PY
import json, pathlib, statistics
out=pathlib.Path('$out'); name='$name'
rows=[json.loads(x) for x in (out/'benchmark.jsonl').read_text().splitlines() if x.strip()] if (out/'benchmark.jsonl').exists() else []
print('SW_SUMMARY', name, 'rows', len(rows))
if rows:
 b=[r['bandwidth_gbs'] for r in rows]; t=[r['time_us'] for r in rows]
 print('gmean_us', statistics.geometric_mean(t), 'gmean_bw', statistics.geometric_mean(b), 'max_bw', max(b))
 by={}
 for r in rows:
  p=r['params']; key=f"wt={p['without_transpose']};npt={p['num_per_tokens']};clamp={p['swiglu_clamp_value']}"
  by.setdefault(key,[]).append(r['bandwidth_gbs'])
 for k,v in sorted(by.items()): print('split', k, len(v), statistics.geometric_mean(v))
PY
}
run_swiglu() {
  name="$1"; shift
  out="$ROOT/swiglu/$name"; mkdir -p "$out"
  echo "=== SW_MORE $name ==="
  timeout 300s env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    /data/venvs/ai-infra/bin/python -m pytest -q tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py::test_swiglu_forward_and_per_channel_cast_and_transpose_benchmark \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose > "$out/pytest.log" 2>&1
  echo $? > "$out/status.txt"; tail -12 "$out/pytest.log" || true; summ_swiglu "$name"
}

run_pc combined_plain512_rescale16 env TK_PC_TILE_K_PLAIN=512 TK_PC_THREADS_PER_TOKEN_RESCALE=16
run_pc combined_plain256_rescale8 env TK_PC_TILE_K_PLAIN=256 TK_PC_THREADS_PER_TOKEN_RESCALE=8
run_pc combined_plain256_rescale4 env TK_PC_TILE_K_PLAIN=256 TK_PC_THREADS_PER_TOKEN_RESCALE=4

run_swiglu t_64x64 env TK_SWIGLU_T_TILE_X=64 TK_SWIGLU_T_TILE_Y=64
run_swiglu t_64x128 env TK_SWIGLU_T_TILE_X=64 TK_SWIGLU_T_TILE_Y=128
run_swiglu t_128x128 env TK_SWIGLU_T_TILE_X=128 TK_SWIGLU_T_TILE_Y=128
