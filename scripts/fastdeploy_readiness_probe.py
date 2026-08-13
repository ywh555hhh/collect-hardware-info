#!/usr/bin/env python3
"""Probe FastDeploy LLM-serving readiness on an accelerator host.

The script intentionally avoids installing packages or mutating the host. It
captures the runtime facts needed to decide whether a machine can run
FastDeploy LLM serving experiments, and where the first compatibility gap is.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


INTERESTING_PACKAGES = [
    "paddlepaddle",
    "paddlepaddle-gpu",
    "paddle-metax-gpu",
    "fastdeploy",
    "paddlenlp",
    "paddleformers",
    "fastapi",
    "uvicorn",
    "openai",
    "transformers",
    "tokenizers",
    "safetensors",
]

INTERESTING_MODULES = [
    "paddle",
    "fastdeploy",
    "paddlenlp",
    "paddleformers",
    "fastapi",
    "uvicorn",
    "openai",
    "transformers",
]


def run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {
        "found": spec is not None,
        "origin": getattr(spec, "origin", None) if spec else None,
    }


def import_probe(module_name: str, attr: str | None = None) -> dict[str, Any]:
    try:
        mod = importlib.import_module(module_name)
        if attr is not None:
            obj: Any = mod
            for part in attr.split("."):
                obj = getattr(obj, part)
            return {"ok": True, "repr": repr(obj)[:500]}
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def paddle_probe(run_check: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"import": import_probe("paddle")}
    if not rec["import"].get("ok"):
        return rec
    import paddle  # type: ignore[import-not-found]

    rec.update(
        {
            "version": getattr(paddle, "__version__", None),
            "device": None,
            "is_compiled_with_cuda": None,
            "is_compiled_with_xpu": None,
            "is_compiled_with_cinn": None,
            "custom_device_types": None,
        }
    )
    for key, expr in [
        ("device", lambda: paddle.device.get_device()),
        ("is_compiled_with_cuda", lambda: paddle.is_compiled_with_cuda()),
        ("is_compiled_with_xpu", lambda: paddle.is_compiled_with_xpu()),
        ("is_compiled_with_cinn", lambda: paddle.is_compiled_with_cinn()),
        ("custom_device_types", lambda: paddle.device.get_all_custom_device_type()),
    ]:
        try:
            rec[key] = expr()
        except Exception as exc:  # noqa: BLE001
            rec[key] = {"error": repr(exc)}
    if run_check:
        try:
            paddle.utils.run_check()
            rec["run_check"] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            rec["run_check"] = {"ok": False, "error": repr(exc)}
    return rec


def readiness_gaps(report: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    packages = report["python"]["packages"]
    modules = report["python"]["modules"]
    paddle = report.get("paddle", {})
    fd_checks = report.get("fastdeploy_import_checks", {})

    if not modules.get("fastdeploy", {}).get("found"):
        gaps.append("fastdeploy Python module is not installed")
    if not modules.get("paddlenlp", {}).get("found") and not modules.get("paddleformers", {}).get("found"):
        gaps.append("Paddle LLM model/tokenizer stack is not installed")
    if not modules.get("fastapi", {}).get("found") or not modules.get("uvicorn", {}).get("found"):
        gaps.append("OpenAI-compatible API server dependencies are missing")
    if not (packages.get("paddle-metax-gpu") or packages.get("paddlepaddle-gpu")):
        gaps.append("No accelerator Paddle package detected")
    if paddle.get("is_compiled_with_cinn") is False:
        gaps.append("Paddle was built without CINN; compiler speedup experiments are out of scope")
    if not fd_checks.get("paddle_jit_marker_unified", {}).get("ok"):
        gaps.append("paddle.jit.marker.unified check failed; official FastDeploy MetaX verification may not pass")
    if not fd_checks.get("fastdeploy_gpu_ops", {}).get("ok"):
        gaps.append("FastDeploy custom GPU operator import check failed")
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-paddle-check", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
        },
        "commands": {},
        "python": {
            "modules": {name: module_status(name) for name in INTERESTING_MODULES},
            "packages": {name: package_version(name) for name in INTERESTING_PACKAGES},
        },
    }

    for tool in ["mx-smi", "nvidia-smi", "ixsmi", "macactl"]:
        path = shutil.which(tool)
        report["commands"][tool] = {"path": path}
        if path:
            report["commands"][tool]["probe"] = run([tool], timeout=25)

    report["paddle"] = paddle_probe(run_check=args.run_paddle_check)
    report["fastdeploy_import_checks"] = {
        "paddle_jit_marker_unified": import_probe("paddle.jit.marker", "unified"),
        "fastdeploy": import_probe("fastdeploy"),
        "fastdeploy_llm": import_probe("fastdeploy", "LLM"),
        "fastdeploy_sampling_params": import_probe("fastdeploy", "SamplingParams"),
        "fastdeploy_openai_api_server": import_probe("fastdeploy.entrypoints.openai.api_server"),
        "fastdeploy_gpu_ops": import_probe("fastdeploy.model_executor.ops.gpu", "beam_search_softmax"),
    }
    report["readiness_gaps"] = readiness_gaps(report)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output_json": str(out), "readiness_gaps": report["readiness_gaps"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
