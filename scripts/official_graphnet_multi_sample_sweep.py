#!/usr/bin/env python3
"""Run a small official GraphNet Paddle sample sweep on C500.

This is a thin orchestrator around official_graphnet_static_patch_probe.py. It
runs the same generated `_C_ops` -> high-level Paddle rewrite benchmark across
multiple official Paddle samples and aggregates success/failure/timing into one
JSON report.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES = [
    "paddle_samples/PaddleNLP/ernie-3.0-nano-zh",
    "paddle_samples/PaddleNLP/ernie-3.0-tiny-pico-v2-zh",
    "paddle_samples/PaddleNLP/ernie-3.0-tiny-base-v2-zh",
    "paddle_samples/PaddleNLP/rocketqa-nano-cross-encoder",
    "paddle_samples/PaddleNLP/uer_chinese-roberta-tiny",
]


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "elapsed_s": round(time.perf_counter() - started, 3), "error": repr(exc)}


def safe_name(sample: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.strip("/"))


def extract_summary(result_json: Path) -> dict[str, Any]:
    rec = json.loads(result_json.read_text())
    parsed = rec.get("parsed", {})
    summary: dict[str, Any] = {
        "ok": rec.get("ok"),
        "benchmark_status_ok": rec.get("benchmark_status_ok"),
        "patch_count": (rec.get("model_patch") or {}).get("count"),
        "status_lines": parsed.get("status_lines", []),
        "performance_lines": parsed.get("performance_lines", []),
        "first_error_lines": parsed.get("first_error_lines", [])[:6],
    }
    for line in parsed.get("performance_lines", []):
        if "[Performance][eager]" in line:
            stats = json.loads(line[line.index("{") :])
            summary["eager_e2e_median_ms"] = stats.get("e2e", {}).get("median")
            summary["eager_e2e_mean_ms"] = stats.get("e2e", {}).get("mean")
            summary["eager_gpu_median_ms"] = stats.get("gpu", {}).get("median")
        if "[Performance][compiled]" in line:
            stats = json.loads(line[line.index("{") :])
            summary["compiled_e2e_median_ms"] = stats.get("e2e", {}).get("median")
            summary["compiled_e2e_mean_ms"] = stats.get("e2e", {}).get("mean")
            summary["compiled_gpu_median_ms"] = stats.get("gpu", {}).get("median")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--runner", default=str(Path(__file__).with_name("official_graphnet_static_patch_probe.py")))
    parser.add_argument("--sample", action="append", dest="samples", default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-per-sample", type=int, default=360)
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = args.samples or DEFAULT_SAMPLES
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": vars(args),
        "samples": samples,
        "results": [],
    }
    for idx, sample in enumerate(samples):
        name = safe_name(sample)
        sample_dir = work_root / f"{idx:02d}_{name}"
        sample_out = out_path.parent / f"{idx:02d}_{name}.json"
        cmd = [
            sys.executable,
            args.runner,
            "--archive",
            args.archive,
            "--work-dir",
            str(sample_dir),
            "--output-json",
            str(sample_out),
            "--sample",
            sample,
            "--warmup",
            str(args.warmup),
            "--trials",
            str(args.trials),
            "--timeout",
            str(args.timeout_per_sample),
        ]
        cmd_result = run_cmd(cmd, args.timeout_per_sample + 30)
        entry: dict[str, Any] = {
            "sample": sample,
            "sample_output_json": str(sample_out),
            "runner_returncode": cmd_result.get("returncode"),
            "runner_elapsed_s": cmd_result.get("elapsed_s"),
        }
        if sample_out.exists():
            entry.update(extract_summary(sample_out))
        else:
            entry.update(
                {
                    "ok": False,
                    "runner_stdout_tail": (cmd_result.get("stdout") or "")[-2000:],
                    "runner_stderr_tail": (cmd_result.get("stderr") or "")[-2000:],
                    "runner_error": cmd_result.get("error"),
                }
            )
        report["results"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if not entry.get("ok") and not args.keep_going:
            break
    ok_count = sum(1 for item in report["results"] if item.get("ok"))
    report["summary"] = {
        "total": len(report["results"]),
        "ok": ok_count,
        "failed": len(report["results"]) - ok_count,
        "success_rate": round(ok_count / len(report["results"]), 4) if report["results"] else 0.0,
    }
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
