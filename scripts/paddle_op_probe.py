#!/usr/bin/env python3
"""Run a small Paddle op compatibility/perf probe for backend triage."""

from __future__ import annotations

import json
import contextlib
import io
import time
import traceback
from typing import Any, Callable

import paddle


def sync() -> None:
    try:
        paddle.device.cuda.synchronize()
    except Exception:
        pass


def bench(name: str, fn: Callable[[], Any], warmup: int = 5, iters: int = 20) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name}
    try:
        paddle.device.set_device("gpu:0")
        y = None
        for _ in range(warmup):
            y = fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn()
        sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / iters
        rec.update(
            {
                "ok": True,
                "iters": iters,
                "avg_ms": round(elapsed_ms, 3),
                "place": str(y.place),
                "shape": list(y.shape),
                "dtype": str(y.dtype),
                "sum": float(y.astype("float32").sum().numpy()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-2000:]})
    return rec


def main() -> None:
    paddle.device.set_device("gpu:0")
    results: list[dict[str, Any]] = []

    x1024 = paddle.randn([1024, 1024], dtype="float32")
    y1024 = paddle.randn([1024, 1024], dtype="float32")
    results.append(bench("matmul_1024_fp32", lambda: x1024 @ y1024))

    try:
        xh = paddle.randn([1024, 1024], dtype="float16")
        yh = paddle.randn([1024, 1024], dtype="float16")
        results.append(bench("matmul_1024_fp16", lambda: xh @ yh))
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "matmul_1024_fp16_setup", "ok": False, "error": repr(exc)})

    xs = paddle.randn([4096, 1024], dtype="float32")
    results.append(bench("softmax_4096x1024", lambda: paddle.nn.functional.softmax(xs, axis=-1)))

    ln = paddle.nn.LayerNorm(1024)
    results.append(bench("layernorm_4096x1024", lambda: ln(xs)))

    conv = paddle.nn.Conv2D(64, 128, 3, padding=1)
    img = paddle.randn([16, 64, 56, 56], dtype="float32")
    results.append(bench("conv2d_16x64x56x56", lambda: conv(img), warmup=3, iters=10))

    base = paddle.randn([200000, 128], dtype="float32")
    idx = paddle.randint(0, 200000, [65536], dtype="int64")
    results.append(bench("gather_65536x128", lambda: paddle.gather(base, idx, axis=0)))

    updates = paddle.randn([65536, 128], dtype="float32")
    out = paddle.zeros([200000, 128], dtype="float32")
    results.append(bench("scatter_65536x128", lambda: paddle.scatter(out, idx, updates, overwrite=False), warmup=2, iters=8))

    extra: dict[str, Any] = {}
    try:
        run_check_log = io.StringIO()
        with contextlib.redirect_stdout(run_check_log), contextlib.redirect_stderr(run_check_log):
            paddle.utils.run_check()
        extra["run_check"] = "ok"
        extra["run_check_log"] = run_check_log.getvalue()
    except Exception as exc:  # noqa: BLE001
        extra["run_check"] = {"error": repr(exc)}

    try:
        from paddle import inference

        extra["inference_config_gpu_related_methods"] = [
            m
            for m in dir(inference.Config)
            if "gpu" in m.lower() or "trt" in m.lower() or "mkldnn" in m.lower()
        ]
    except Exception as exc:  # noqa: BLE001
        extra["inference_error"] = repr(exc)

    print(
        json.dumps(
            {
                "paddle_version": paddle.__version__,
                "device": paddle.device.get_device(),
                "results": results,
                "extra": extra,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
