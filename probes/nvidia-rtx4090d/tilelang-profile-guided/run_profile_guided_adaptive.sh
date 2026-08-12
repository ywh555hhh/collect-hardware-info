#!/usr/bin/env bash
set -uo pipefail

cd /data/src/TileKernels-Metax || exit 1
ROOT=/data/results/tileops-4090/profile_guided_adaptive_20260811
mkdir -p "$ROOT"

run_one() {
  local op="$1"; shift
  local test="$1"; shift
  local out="$ROOT/$op"
  mkdir -p "$out"
  echo "=== ADAPTIVE $op ==="
  env PYTHONPATH=/data/src/TileKernels-Metax TILELANG_PRINT_ON_COMPILATION=0 \
    /data/venvs/ai-infra/bin/python -m pytest -q "$test" \
    -m benchmark --run-benchmark --benchmark-output "$out/benchmark.jsonl" --benchmark-verbose \
    > "$out/pytest.log" 2>&1
  local status=$?
  echo "$status" > "$out/status.txt"
  tail -20 "$out/pytest.log" || true
  /data/venvs/ai-infra/bin/python - <<PY
import json, pathlib, statistics, math
op = "$op"
out = pathlib.Path("$out")
rows = [json.loads(x) for x in (out / "benchmark.jsonl").read_text().splitlines() if x.strip()] if (out / "benchmark.jsonl").exists() else []
print("ADAPTIVE_SUMMARY", op, "status", "$status", "rows", len(rows))
if rows:
    b = [r["bandwidth_gbs"] for r in rows if r.get("bandwidth_gbs", 0) > 0]
    t = [r["time_us"] for r in rows if r.get("time_us", 0) > 0]
    print("gmean_bw", statistics.geometric_mean(b), "max_bw", max(b), "gmean_us", statistics.geometric_mean(t))
    by = {}
    for r in rows:
        p = r.get("params", {})
        keys = []
        for k in ("dtype", "without_transpose", "num_per_tokens", "is_fused_cast_back", "num_topk", "num_ep_ranks"):
            if k in p:
                keys.append(f"{k}={p[k]}")
        key = ";".join(keys) if keys else "all"
        by.setdefault(key, []).append(r["bandwidth_gbs"])
    for key, vals in sorted(by.items())[:80]:
        if len(vals) >= 2:
            print("split", key, len(vals), statistics.geometric_mean(vals))
PY
}

run_one per_channel tests/quant/test_per_channel_cast_fused.py::test_per_channel_cast_fused_benchmark
run_one batched_transpose tests/transpose/test_transpose.py::test_batched_transpose_benchmark
run_one swiglu tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py::test_swiglu_forward_and_per_channel_cast_and_transpose_benchmark

/data/venvs/ai-infra/bin/python - <<'PY'
import json, pathlib, statistics, math
root = pathlib.Path("/data/results/tileops-4090/profile_guided_adaptive_20260811")
baselines = {
  "per_channel": pathlib.Path("/data/results/tileops-4090/opt_sweep/per_channel_cast_fused_20260811_repeat1/baseline_warp32/benchmark.jsonl"),
  "batched_transpose": pathlib.Path("/data/results/tileops-4090/three_ops_push_20260811/batched_transpose/baseline/benchmark.jsonl"),
  "swiglu": pathlib.Path("/data/results/tileops-4090/three_ops_push_20260811/swiglu/baseline/benchmark.jsonl"),
}
def load(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[json.dumps(r["params"], sort_keys=True, separators=(",", ":"))] = r
    return out
for op, base_path in baselines.items():
    cand_path = root / op / "benchmark.jsonl"
    A, B = load(base_path), load(cand_path)
    rows = []
    for k, r0 in A.items():
        if k in B and B[k].get("time_us", 0):
            rows.append((r0["time_us"] / B[k]["time_us"], r0, B[k]))
    print("COMPARE", op, "matched", len(rows))
    if rows:
        g = math.exp(sum(math.log(x[0]) for x in rows) / len(rows))
        print("COMPARE_GMEAN_SPEEDUP", op, g)
        rows.sort(key=lambda x: x[0], reverse=True)
        print("TOP_GAIN", rows[0][0], rows[0][1]["params"], rows[0][1]["time_us"], rows[0][2]["time_us"])
        print("TOP_LOSS", rows[-1][0], rows[-1][1]["params"], rows[-1][1]["time_us"], rows[-1][2]["time_us"])
PY
