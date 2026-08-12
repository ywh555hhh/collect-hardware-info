#!/usr/bin/env python3
"""Try a temporary generated-code patch for official GraphNet Paddle static replay.

The first official bring-up showed:
- direct dygraph forward for ernie-3.0-nano-zh succeeds on C500,
- official benchmark static/eager path fails under paddle.jit.to_static around
  generated paddle._C_ops.full calls.

This probe unpacks GraphNet into a temporary workdir, applies the backend import
compatibility patch, rewrites only the selected sample's generated `model.py`
`paddle._C_ops.full(...)` calls into high-level `paddle.full(...)`, then runs the
official `compiler=nope` benchmark again. It records whether this bypass is
sufficient or which next operator becomes the blocker.
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


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False, timeout=timeout)
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "cwd": str(cwd), "elapsed_s": round(time.perf_counter() - started, 3), "error": repr(exc)}


def shorten(result: dict[str, Any], limit: int = 10000) -> dict[str, Any]:
    out = dict(result)
    for key in ("stdout", "stderr"):
        if isinstance(out.get(key), str) and len(out[key]) > limit:
            out[key] = out[key][-limit:]
    return out


def find_root(work_dir: Path) -> Path:
    for child in work_dir.iterdir():
        if child.is_dir() and (child / "graph_net").is_dir() and (child / "graph_net_bench").is_dir():
            return child
    raise RuntimeError(f"GraphNet root not found under {work_dir}")


def apply_backend_import_compat(root: Path) -> dict[str, Any]:
    src = root / "graph_net_bench" / "paddle" / "backend"
    dst = root / "graph_net" / "paddle" / "backend"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for py_file in src.glob("*.py"):
        shutil.copy2(py_file, dst / py_file.name)
        copied.append(py_file.name)
    (dst / "__init__.py").touch()
    return {"ok": True, "copied": sorted(copied)}


def split_top_level_args(arg_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(arg_text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(arg_text[start:idx].strip())
            start = idx + 1
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def dtype_token_to_string(dtype_expr: str) -> str:
    expr = dtype_expr.strip()
    mapping = {
        "paddle.int64": "int64",
        "paddle.int32": "int32",
        "paddle.float32": "float32",
        "paddle.bool": "bool",
    }
    return mapping.get(expr, expr.rsplit(".", 1)[-1].strip("\"'"))


def find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    in_string: str | None = None
    escape = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in {"'", '"'}:
            in_string = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def rewrite_cops_call(text: str, op_name: str, converter) -> tuple[str, list[dict[str, str]]]:
    marker = f"paddle._C_ops.{op_name}("
    replacements: list[dict[str, str]] = []
    out_parts: list[str] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            out_parts.append(text[cursor:])
            break
        open_idx = start + len(marker) - 1
        end = find_matching_paren(text, open_idx)
        if end < 0:
            out_parts.append(text[cursor:])
            break
        call = text[start : end + 1]
        args = split_top_level_args(text[open_idx + 1 : end])
        out_parts.append(text[cursor:start])
        replacement = converter(args)
        if replacement:
            out_parts.append(replacement)
            replacements.append({"original": call, "replacement": replacement})
        else:
            out_parts.append(call)
        cursor = end + 1
    return "".join(out_parts), replacements


def patch_cops_full(model_path: Path) -> dict[str, Any]:
    text = model_path.read_text()
    model_path.with_suffix(".model_py_before_cops_patch").write_text(text)
    all_replacements: list[dict[str, str]] = []

    def full_converter(args: list[str]) -> str | None:
        if len(args) != 4:
            return None
        shape, value, dtype_expr, _place = args
        dtype = dtype_token_to_string(dtype_expr)
        return f'paddle.full({shape}, {value}, dtype="{dtype}")'

    def equal_converter(args: list[str]) -> str | None:
        if len(args) != 2:
            return None
        return f"paddle.equal({args[0]}, {args[1]})"

    def cast_converter(args: list[str]) -> str | None:
        if len(args) != 2:
            return None
        dtype = dtype_token_to_string(args[1])
        return f'paddle.cast({args[0]}, dtype="{dtype}")'

    def scale_converter(args: list[str]) -> str | None:
        if len(args) != 4:
            return None
        x, scale, bias, bias_after_scale = args
        if bias.strip() == 'float("0")' and bias_after_scale.strip() == "True":
            return f"({x} * {scale})"
        return f"paddle.scale({x}, scale={scale}, bias={bias}, bias_after_scale={bias_after_scale})"

    for op_name, converter in [("full", full_converter), ("equal", equal_converter), ("cast", cast_converter), ("scale", scale_converter)]:
        text, replacements = rewrite_cops_call(text, op_name, converter)
        all_replacements.extend({"op": op_name, **item} for item in replacements)

    model_path.write_text(text)
    return {"ok": True, "count": len(all_replacements), "replacements": all_replacements[:30]}

def parse_log(text: str) -> dict[str, Any]:
    parsed = {"status_lines": [], "performance_lines": [], "config_lines": [], "first_error_lines": []}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "[Result]" in line:
            parsed["status_lines"].append(line)
        if "[Performance]" in line:
            parsed["performance_lines"].append(line)
        if "[Config]" in line or "[Processing]" in line:
            parsed["config_lines"].append(line)
        if "failed:" in line or "ValueError:" in line or "RuntimeError:" in line:
            parsed["first_error_lines"].append(line)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample", default="paddle_samples/PaddleNLP/ernie-3.0-nano-zh")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "params": vars(args),
    }
    try:
        with tarfile.open(args.archive, "r:gz") as tf:
            tf.extractall(work_dir)
        root = find_root(work_dir)
        sample_path = root / args.sample
        rec["graphnet_root"] = str(root)
        rec["sample_path"] = str(sample_path)
        rec["backend_patch"] = apply_backend_import_compat(root)
        rec["model_patch"] = patch_cops_full(sample_path / "model.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
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
                "c500-official-graphnet-static-patch-log",
            ],
            cwd=root,
            env=env,
            timeout=args.timeout,
        )
        prefix = out_path.parent / "official_graphnet_static_patch_ernie_nano_nope"
        stdout_path = prefix.with_suffix(".stdout.log")
        stderr_path = prefix.with_suffix(".stderr.log")
        stdout_path.write_text(benchmark.get("stdout", ""))
        stderr_path.write_text(benchmark.get("stderr", ""))
        rec["benchmark"] = shorten(benchmark)
        rec["stdout_log"] = str(stdout_path)
        rec["stderr_log"] = str(stderr_path)
        rec["parsed"] = parse_log((benchmark.get("stdout") or "") + "\n" + (benchmark.get("stderr") or ""))
        status_text = "\n".join(rec["parsed"].get("status_lines", []))
        rec["benchmark_status_ok"] = bool(status_text) and "failed" not in status_text.lower()
        rec["ok"] = benchmark.get("returncode") == 0 and rec["benchmark_status_ok"]
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(exc)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({"ok": rec.get("ok"), "benchmark_status_ok": rec.get("benchmark_status_ok"), "parsed": rec.get("parsed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
