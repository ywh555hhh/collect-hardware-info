#!/usr/bin/env python3
"""Probe Paddle device aliases on the C500 Paddle image."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import paddle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--devices",
        nargs="*",
        default=["cuda", "cuda:0", "gpu", "gpu:0", "maca", "maca:0", "cpu"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "paddle_version": paddle.__version__,
        "results": [],
    }
    for device in args.devices:
        item: dict[str, Any] = {"device": device}
        try:
            paddle.device.set_device(device)
            x = paddle.randn((16, 16), dtype="float32")
            y = paddle.matmul(x, x)
            try:
                paddle.device.cuda.synchronize()
            except Exception:
                pass
            item.update(
                {
                    "ok": True,
                    "current_device": paddle.device.get_device(),
                    "place": str(y.place),
                    "sum": float(y.sum().numpy()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]})
        rec["results"].append(item)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
