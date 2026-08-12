#!/usr/bin/env python3
"""Mini GraphNet-style Paddle workload probe for C500.

This is not the official PaddlePaddle/GraphNet benchmark. It is a controlled
computation-graph workload approximation focused on compiler/inference-relevant
operator families: dense matmul, normalization, gather/scatter, and mixed graph
blocks. The goal is to characterize what the current C500 Paddle image can run
and where static/Paddle Inference follow-up is viable.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import paddle


def sync() -> None:
    try:
        paddle.device.cuda.synchronize()
    except Exception:
        pass


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-5000:],
            "stderr": proc.stderr[-2000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


class MxSmiSampler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def snapshot(self, label: str) -> dict[str, Any]:
        if not self.enabled:
            return {"label": label, "enabled": False}
        return {
            "label": label,
            "mx_smi": run_cmd(["mx-smi"], timeout=10),
            "usage": run_cmd(["mx-smi", "--show-usage"], timeout=10),
            "hbm_bandwidth": run_cmd(["mx-smi", "--show-hbm-bandwidth"], timeout=10),
        }


def bench(name: str, fn: Callable[[], paddle.Tensor], warmup: int, iters: int) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name, "warmup": warmup, "iters": iters}
    try:
        y = None
        for _ in range(warmup):
            y = fn()
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = fn()
            sync()
            times.append((time.perf_counter() - t0) * 1000)
        arr = sorted(times)
        rec.update(
            {
                "ok": True,
                "avg_ms": round(sum(times) / len(times), 4),
                "min_ms": round(arr[0], 4),
                "p50_ms": round(arr[len(arr) // 2], 4),
                "p90_ms": round(arr[min(len(arr) - 1, int(round((len(arr) - 1) * 0.9)))], 4),
                "max_ms": round(arr[-1], 4),
                "place": str(y.place),
                "shape": list(y.shape),
                "dtype": str(y.dtype),
                "sum": float(y.astype("float32").sum().numpy()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-4000:]})
    return rec


class DenseBlock(paddle.nn.Layer):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.w1 = self.create_parameter([hidden, hidden], dtype="float32")
        self.w2 = self.create_parameter([hidden, hidden], dtype="float32")
        self.norm = paddle.nn.LayerNorm(hidden)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y = paddle.matmul(x, self.w1)
        y = paddle.nn.functional.gelu(y)
        y = paddle.matmul(y, self.w2)
        return self.norm(y + x)


class SparseBlock(paddle.nn.Layer):
    def __init__(self, nodes: int, hidden: int) -> None:
        super().__init__()
        self.nodes = nodes
        self.hidden = hidden
        self.scale = self.create_parameter([hidden], dtype="float32")

    def forward(self, x: paddle.Tensor, src: paddle.Tensor, dst: paddle.Tensor) -> paddle.Tensor:
        msg = paddle.gather(x, src, axis=0) * self.scale
        out = paddle.zeros([self.nodes, self.hidden], dtype=x.dtype)
        return paddle.scatter(out, dst, msg, overwrite=False)


class MixedGraphBlock(paddle.nn.Layer):
    def __init__(self, nodes: int, hidden: int) -> None:
        super().__init__()
        self.dense = DenseBlock(hidden)
        self.sparse = SparseBlock(nodes, hidden)
        self.proj = self.create_parameter([hidden, hidden], dtype="float32")

    def forward(self, x: paddle.Tensor, src: paddle.Tensor, dst: paddle.Tensor) -> paddle.Tensor:
        y = self.dense(x)
        s = self.sparse(y, src, dst)
        return paddle.matmul(paddle.nn.functional.relu(y + s), self.proj)


def try_jit_export(layer: paddle.nn.Layer, input_specs: list[paddle.static.InputSpec], sample_args: tuple[Any, ...]) -> dict[str, Any]:
    rec: dict[str, Any] = {"attempted": True}
    tmp = Path(tempfile.mkdtemp(prefix="paddle_graphnet_jit_"))
    try:
        layer.eval()
        static_layer = paddle.jit.to_static(layer, input_spec=input_specs)
        save_prefix = str(tmp / "model")
        paddle.jit.save(static_layer, save_prefix)
        rec["save_ok"] = True
        rec["files"] = sorted(p.name for p in tmp.iterdir())
        loaded = paddle.jit.load(save_prefix)
        loaded.eval()
        for _ in range(3):
            out = loaded(*sample_args)
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            out = loaded(*sample_args)
        sync()
        rec.update(
            {
                "load_and_run_ok": True,
                "avg_ms": round((time.perf_counter() - t0) * 1000 / 20, 4),
                "place": str(out.place),
                "shape": list(out.shape),
                "sum": float(out.astype("float32").sum().numpy()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-5000:]})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sample-mxsmi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": vars(args),
        "python": sys.version,
        "paddle_version": paddle.__version__,
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("PADDLE", "FLAGS", "MACA", "LD_"))},
    }
    sampler = MxSmiSampler(args.sample_mxsmi)
    try:
        paddle.device.set_device(args.device)
        rec["device"] = paddle.device.get_device()
        rec["mxsmi_before"] = sampler.snapshot("before")

        x = paddle.randn([args.nodes, args.hidden], dtype="float32")
        src = paddle.randint(0, args.nodes, [args.edges], dtype="int64")
        dst = paddle.randint(0, args.nodes, [args.edges], dtype="int64")
        dense = DenseBlock(args.hidden)
        sparse = SparseBlock(args.nodes, args.hidden)
        mixed = MixedGraphBlock(args.nodes, args.hidden)
        dense.eval()
        sparse.eval()
        mixed.eval()

        results = []
        results.append(bench("dense_block", lambda: dense(x), args.warmup, args.iters))
        results.append(bench("sparse_gather_scatter_block", lambda: sparse(x, src, dst), args.warmup, args.iters))
        results.append(bench("mixed_graph_block", lambda: mixed(x, src, dst), args.warmup, args.iters))
        results.append(bench("graph_readout_reduce", lambda: mixed(x, src, dst).mean(axis=0), args.warmup, args.iters))
        rec["dynamic_results"] = results

        rec["jit_export"] = {
            "dense_block": try_jit_export(
                dense,
                [paddle.static.InputSpec(shape=[None, args.hidden], dtype="float32", name="x")],
                (x,),
            ),
            "mixed_graph_block": try_jit_export(
                mixed,
                [
                    paddle.static.InputSpec(shape=[None, args.hidden], dtype="float32", name="x"),
                    paddle.static.InputSpec(shape=[None], dtype="int64", name="src"),
                    paddle.static.InputSpec(shape=[None], dtype="int64", name="dst"),
                ],
                (x, src, dst),
            ),
        }
        try:
            from paddle import inference

            rec["paddle_inference_api"] = {
                "available": True,
                "gpu_methods": [m for m in dir(inference.Config) if "gpu" in m.lower() or "trt" in m.lower()],
            }
        except Exception as exc:  # noqa: BLE001
            rec["paddle_inference_api"] = {"available": False, "error": repr(exc)}
        rec["mxsmi_after"] = sampler.snapshot("after")
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()[-8000:]
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({"ok": rec.get("ok"), "dynamic_results": rec.get("dynamic_results"), "jit_export": rec.get("jit_export")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
