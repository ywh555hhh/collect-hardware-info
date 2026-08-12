#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811_repeat1
mkdir -p "$ROOT"
cd /data/src/TileKernels-Metax || exit 1

run_cfg() {
  name="$1"
  shift
  out="$ROOT/$name"
  mkdir -p "$out"
  echo "=== REPEAT $name ==="
  echo "$*" > "$out/env.txt"
  env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 "$@" \
    /data/venvs/ai-infra/bin/python -m pytest -q tests/quant/test_per_channel_cast_fused.py::test_per_channel_cast_fused_benchmark \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose > "$out/pytest.log" 2>&1
  status=$?
  echo "$status" > "$out/status.txt"
  tail -20 "$out/pytest.log" || true
  /data/venvs/ai-infra/bin/python - <<PY
import json, pathlib, statistics
out = pathlib.Path("$out")
rows = [json.loads(x) for x in (out / "benchmark.jsonl").read_text().splitlines() if x.strip()] if (out / "benchmark.jsonl").exists() else []
print("SUMMARY", "$name", "status", "$status", "rows", len(rows))
if rows:
    t = [r["time_us"] for r in rows]
    b = [r["bandwidth_gbs"] for r in rows if r.get("bandwidth_gbs", 0) > 0]
    plain = [r["bandwidth_gbs"] for r in rows if not r["params"]["is_fused_cast_back"]]
    rescale = [r["bandwidth_gbs"] for r in rows if r["params"]["is_fused_cast_back"]]
    print("gmean_us", statistics.geometric_mean(t), "gmean_bw", statistics.geometric_mean(b), "max_bw", max(b))
    print("plain_bw", statistics.geometric_mean(plain), "rescale_bw", statistics.geometric_mean(rescale))
PY
}

run_cfg baseline_warp32 env
run_cfg plain_tile256 env TK_PC_TILE_K_PLAIN=256
run_cfg rescale_tpt16 env TK_PC_THREADS_PER_TOKEN_RESCALE=16
run_cfg rescale_tpt32 env TK_PC_THREADS_PER_TOKEN_RESCALE=32

/data/venvs/ai-infra/bin/python /data/results/tileops-4090/opt_sweep/summarize_pc_sweep.py "$ROOT" || true
