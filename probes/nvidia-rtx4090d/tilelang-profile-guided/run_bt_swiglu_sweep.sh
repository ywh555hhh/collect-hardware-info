#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/results/tileops-4090/three_ops_push_20260811
mkdir -p "$ROOT"
cd /data/src/TileKernels-Metax || exit 1

summarize_jsonl() {
  local name="$1" out="$2" status="$3"
  /data/venvs/ai-infra/bin/python - <<PY
import json, pathlib, statistics
name='$name'; out=pathlib.Path('$out'); status='$status'
rows=[json.loads(x) for x in (out/'benchmark.jsonl').read_text().splitlines() if x.strip()] if (out/'benchmark.jsonl').exists() else []
print('SUMMARY', name, 'status', status, 'rows', len(rows))
if rows:
    b=[r['bandwidth_gbs'] for r in rows if r.get('bandwidth_gbs',0)>0]
    t=[r['time_us'] for r in rows]
    print('gmean_us', statistics.geometric_mean(t), 'gmean_bw', statistics.geometric_mean(b), 'max_bw', max(b))
    # useful path splits
    by={}
    for r in rows:
        p=r.get('params',{})
        key=[]
        for k in ('dtype','without_transpose','num_per_tokens','swiglu_clamp_value'):
            if k in p: key.append(f'{k}={p[k]}')
        key=';'.join(key) if key else 'all'
        by.setdefault(key, []).append(r['bandwidth_gbs'])
    for key, vals in sorted(by.items())[:40]:
        if len(vals) >= 2:
            print('split', key, len(vals), statistics.geometric_mean(vals))
PY
}

run_batched() {
  local name="$1"; shift
  local out="$ROOT/batched_transpose/$name"; mkdir -p "$out"
  echo "=== BT $name ==="; echo "$*" > "$out/env.txt"
  env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    /data/venvs/ai-infra/bin/python -m pytest -q tests/transpose/test_transpose.py::test_batched_transpose_benchmark \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose > "$out/pytest.log" 2>&1
  status=$?; echo "$status" > "$out/status.txt"; tail -12 "$out/pytest.log" || true; summarize_jsonl "bt/$name" "$out" "$status"
}

run_swiglu() {
  local name="$1"; shift
  local out="$ROOT/swiglu/$name"; mkdir -p "$out"
  echo "=== SWIGLU $name ==="; echo "$*" > "$out/env.txt"
  env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    /data/venvs/ai-infra/bin/python -m pytest -q tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py::test_swiglu_forward_and_per_channel_cast_and_transpose_benchmark \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose > "$out/pytest.log" 2>&1
  status=$?; echo "$status" > "$out/status.txt"; tail -12 "$out/pytest.log" || true; summarize_jsonl "swiglu/$name" "$out" "$status"
}

run_batched baseline env
run_batched block_k2 env TK_BT_BLOCK_K=2
run_batched block_k8 env TK_BT_BLOCK_K=8
run_batched threads128 env TK_BT_NUM_THREADS=128
run_batched threads512 env TK_BT_NUM_THREADS=512
run_batched block_x64 env TK_BT_BLOCK_X=64
run_batched block_y64 env TK_BT_BLOCK_Y=64
run_batched block_64x64 env TK_BT_BLOCK_X=64 TK_BT_BLOCK_Y=64

run_swiglu baseline env
run_swiglu threads128 env TK_SWIGLU_NUM_THREADS=128
run_swiglu threads512 env TK_SWIGLU_NUM_THREADS=512
run_swiglu tile_k2 env TK_SWIGLU_TILE_K=2
run_swiglu tile_k8 env TK_SWIGLU_TILE_K=8
run_swiglu force_128x64 env TK_SWIGLU_TILE_X=128 TK_SWIGLU_TILE_Y=64
run_swiglu nt_128x64 env TK_SWIGLU_NT_TILE_X=128 TK_SWIGLU_NT_TILE_Y=64
run_swiglu t_128x64 env TK_SWIGLU_T_TILE_X=128 TK_SWIGLU_T_TILE_Y=64
