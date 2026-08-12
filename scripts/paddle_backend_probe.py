#!/usr/bin/env python3
"""Probe PaddlePaddle backend/device capabilities on an accelerator image."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def call(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def smoke_on_device(paddle: Any, device: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"device": device}
    try:
        paddle.device.set_device(device)
        x = paddle.randn([256, 256], dtype="float32")
        y = paddle.randn([256, 256], dtype="float32")
        t0 = time.perf_counter()
        z = x @ y
        loss = z.square().mean()
        loss.backward()
        # Force sync through host copy.
        value = float(loss.numpy())
        rec.update(
            {
                "ok": True,
                "place": str(z.place),
                "loss": value,
                "wall_ms": round((time.perf_counter() - t0) * 1000, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc)[:2000]})
    return rec


def main() -> None:
    rec: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "env": {
            k: os.environ.get(k)
            for k in sorted(os.environ)
            if k.startswith(("PADDLE", "FLAGS", "MACA", "LD_", "CUDA", "XPU", "CUSTOM"))
        },
        "commands": {
            "mx_smi": run_cmd(["mx-smi"], timeout=15),
            "pip_paddle": run_cmd([sys.executable, "-m", "pip", "list"], timeout=30),
        },
    }

    try:
        import paddle

        rec["paddle"] = {
            "version": getattr(paddle, "__version__", None),
            "file": getattr(paddle, "__file__", None),
            "current_device": None,
            "available_device": None,
            "all_custom_device_type": None,
            "compiled": {},
        }
        rec["paddle"]["current_device"] = call(paddle.device, "get_device")
        rec["paddle"]["available_device"] = call(paddle.device, "get_available_device")
        rec["paddle"]["all_custom_device_type"] = call(paddle.device, "get_all_custom_device_type")
        for name in [
            "is_compiled_with_cuda",
            "is_compiled_with_xpu",
            "is_compiled_with_rocm",
            "is_compiled_with_custom_device",
            "is_compiled_with_cinn",
        ]:
            rec["paddle"]["compiled"][name] = call(paddle, name)

        try:
            import paddle.base.core as core

            rec["paddle_core"] = {
                "compiled_cuda": call(core, "is_compiled_with_cuda"),
                "compiled_xpu": call(core, "is_compiled_with_xpu"),
                "compiled_custom": call(core, "is_compiled_with_custom_device"),
                "compiled_cinn": call(core, "is_compiled_with_cinn"),
                "cuda_device_count": call(core, "get_cuda_device_count"),
                "all_custom_device_type": call(core, "get_all_custom_device_type"),
            }
        except Exception as exc:  # noqa: BLE001
            rec["paddle_core_error"] = repr(exc)

        rec["smoke_tests"] = [
            smoke_on_device(paddle, dev)
            for dev in ["cpu", "gpu", "gpu:0", "maca", "maca:0", "xpu", "custom_cpu"]
        ]

        try:
            rec["static_graph"] = {}
            paddle.enable_static()
            rec["static_graph"]["enabled"] = True
            paddle.disable_static()
        except Exception as exc:  # noqa: BLE001
            rec["static_graph"] = {"enabled": False, "error": repr(exc)}

        try:
            from paddle import inference

            rec["inference"] = {
                "available": True,
                "Config": str(getattr(inference, "Config", None)),
                "Predictor": str(getattr(inference, "Predictor", None)),
            }
        except Exception as exc:  # noqa: BLE001
            rec["inference"] = {"available": False, "error": repr(exc)}
    except Exception as exc:  # noqa: BLE001
        rec["paddle_import_error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()

    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
