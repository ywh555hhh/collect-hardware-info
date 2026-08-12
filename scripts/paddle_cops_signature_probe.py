#!/usr/bin/env python3
"""Probe low-level Paddle _C_ops signatures relevant to official GraphNet samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import paddle


def try_call(label: str, fn) -> dict[str, Any]:
    rec: dict[str, Any] = {"label": label}
    try:
        out = fn()
        rec.update(
            {
                "ok": True,
                "dtype": str(out.dtype),
                "place": str(out.place),
                "shape": list(out.shape),
                "value": out.numpy().tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]})
    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "paddle_version": paddle.__version__,
        "device_arg": args.device,
        "results": [],
    }
    try:
        paddle.device.set_device(args.device)
    except Exception as exc:  # noqa: BLE001
        rec["set_device_error"] = repr(exc)
    place = paddle.framework._current_expected_place()
    rec["current_device"] = paddle.device.get_device()
    rec["current_expected_place"] = str(place)

    candidates = [
        ("_C_ops.full dtype=paddle.int64", lambda: paddle._C_ops.full([], float("0"), paddle.int64, place)),
        ("_C_ops.full dtype='int64'", lambda: paddle._C_ops.full([], float("0"), "int64", place)),
        (
            "_C_ops.full dtype=VarType.INT64",
            lambda: paddle._C_ops.full([], float("0"), paddle.base.core.VarDesc.VarType.INT64, place),
        ),
        ("paddle.full dtype='int64'", lambda: paddle.full([], 0, dtype="int64")),
        ("paddle.to_tensor dtype='int64'", lambda: paddle.to_tensor(0, dtype="int64")),
        ("_C_ops.cast dtype=paddle.float32", lambda: paddle._C_ops.cast(paddle.to_tensor([True]), paddle.float32)),
        ("_C_ops.cast dtype='float32'", lambda: paddle._C_ops.cast(paddle.to_tensor([True]), "float32")),
        (
            "_C_ops.cast dtype=VarType.FP32",
            lambda: paddle._C_ops.cast(paddle.to_tensor([True]), paddle.base.core.VarDesc.VarType.FP32),
        ),
    ]
    for label, fn in candidates:
        rec["results"].append(try_call(label, fn))

    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
