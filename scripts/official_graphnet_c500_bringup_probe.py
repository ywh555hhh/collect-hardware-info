#!/usr/bin/env python3
"""Bring up the official PaddlePaddle/GraphNet benchmark on a C500 Paddle image.

The probe is intentionally conservative:

1. unpack a provided official GraphNet source archive,
2. try importing the official Paddle benchmark as-is,
3. apply the smallest observed compatibility patch for the current upstream tree,
4. run one real Paddle sample with compiler=nope on the C500 Paddle image,
5. save JSON plus raw stdout/stderr logs.

It does not claim CINN benchmarking. The current C500 image reports
is_compiled_with_cinn() == False, so compiler=nope is the honest first target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else "",
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else "",
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def short_cmd_result(result: dict[str, Any], limit: int = 8000) -> dict[str, Any]:
    out = dict(result)
    for key in ("stdout", "stderr"):
        if isinstance(out.get(key), str) and len(out[key]) > limit:
            out[key] = out[key][-limit:]
    return out


def find_graphnet_root(work_dir: Path) -> Path:
    candidates = []
    for child in work_dir.iterdir():
        if child.is_dir() and (child / "graph_net").is_dir() and (child / "graph_net_bench").is_dir():
            candidates.append(child)
    if not candidates:
        raise RuntimeError(f"Could not find GraphNet root under {work_dir}")
    return candidates[0]


def apply_backend_import_compat(root: Path) -> dict[str, Any]:
    src = root / "graph_net_bench" / "paddle" / "backend"
    dst = root / "graph_net" / "paddle" / "backend"
    rec: dict[str, Any] = {"src": str(src), "dst": str(dst)}
    if not src.is_dir():
        rec.update({"ok": False, "error": "source_backend_dir_missing"})
        return rec
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for py_file in src.glob("*.py"):
        shutil.copy2(py_file, dst / py_file.name)
        copied.append(py_file.name)
    (dst / "__init__.py").touch()
    rec.update({"ok": True, "copied": sorted(copied)})
    return rec


def parse_graphnet_log(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"status_lines": [], "performance_lines": [], "config_lines": []}
    for line in text.splitlines():
        if "[Result]" in line:
            parsed["status_lines"].append(line)
        if "[Performance]" in line:
            parsed["performance_lines"].append(line)
        if "[Config]" in line or "[Processing]" in line:
            parsed["config_lines"].append(line)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample", default="paddle_samples/PaddleNLP/ernie-3.0-nano-zh")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "params": vars(args),
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("PADDLE", "FLAGS", "MACA", "LD_"))},
    }
    try:
        with tarfile.open(args.archive, "r:gz") as tf:
            tf.extractall(work_dir)
        root = find_graphnet_root(work_dir)
        rec["graphnet_root"] = str(root)
        sample_path = root / args.sample
        rec["sample_path"] = str(sample_path)
        rec["sample_exists"] = sample_path.is_dir()
        rec["sample_files"] = sorted(p.name for p in sample_path.iterdir()) if sample_path.is_dir() else []

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        raw_import = run_cmd(
            [
                sys.executable,
                "-c",
                "import graph_net_bench.paddle.test_compiler as tc; print(tc.__file__)",
            ],
            cwd=root,
            env=env,
            timeout=60,
        )
        rec["raw_import"] = short_cmd_result(raw_import)

        rec["compat_patch"] = apply_backend_import_compat(root)
        patched_import = run_cmd(
            [
                sys.executable,
                "-c",
                "import graph_net_bench.paddle.test_compiler as tc; print(tc.__file__)",
            ],
            cwd=root,
            env=env,
            timeout=60,
        )
        rec["patched_import"] = short_cmd_result(patched_import)

        direct_code = f"""
import json
import paddle
from graph_net_bench.paddle import test_compiler
sample = {str(sample_path)!r}
paddle.device.set_device("gpu:0")
input_dict = test_compiler.get_input_dict(sample)
model = test_compiler.get_model(sample)
model.eval()
out = model(**input_dict)
if isinstance(out, paddle.Tensor):
    outs = [out]
elif isinstance(out, (tuple, list)):
    outs = list(out)
else:
    outs = [out]
summary = []
for x in outs:
    if isinstance(x, paddle.Tensor):
        summary.append({{"shape": list(x.shape), "dtype": str(x.dtype), "place": str(x.place), "sum": float(x.astype("float32").sum().numpy())}})
    else:
        summary.append({{"type": type(x).__name__, "repr": repr(x)[:200]}})
print(json.dumps({{"ok": True, "num_outputs": len(outs), "outputs": summary}}, ensure_ascii=False))
"""
        direct_dygraph = run_cmd(
            [sys.executable, "-c", direct_code],
            cwd=root,
            env=env,
            timeout=args.timeout,
        )
        rec["direct_dygraph_run"] = short_cmd_result(direct_dygraph)

        run_log_prefix = output_json.parent / "official_graphnet_ernie_nano_nope"
        benchmark = run_cmd(
            [
                sys.executable,
                "-m",
                "graph_net_bench.paddle.test_compiler",
                "--model-path",
                str(sample_path),
                "--compiler",
                "nope",
                "--device",
                "cuda",
                "--warmup",
                str(args.warmup),
                "--trials",
                str(args.trials),
                "--log-prompt",
                "c500-official-graphnet-log",
            ],
            cwd=root,
            env=env,
            timeout=args.timeout,
        )
        stdout_path = run_log_prefix.with_suffix(".stdout.log")
        stderr_path = run_log_prefix.with_suffix(".stderr.log")
        stdout_path.write_text(benchmark.get("stdout", ""))
        stderr_path.write_text(benchmark.get("stderr", ""))
        rec["benchmark"] = short_cmd_result(benchmark)
        rec["benchmark_stdout_log"] = str(stdout_path)
        rec["benchmark_stderr_log"] = str(stderr_path)
        rec["benchmark_parsed"] = parse_graphnet_log((benchmark.get("stdout") or "") + "\n" + (benchmark.get("stderr") or ""))
        rec["ok"] = benchmark.get("returncode") == 0
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(exc)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({
        "ok": rec.get("ok"),
        "benchmark_status_ok": rec.get("benchmark_status_ok"),
        "raw_import": rec.get("raw_import"),
        "patched_import": rec.get("patched_import"),
        "direct_dygraph_run": rec.get("direct_dygraph_run"),
        "benchmark_parsed": rec.get("benchmark_parsed"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
