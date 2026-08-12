#!/usr/bin/env python3
"""Paddle Inference predictor probe for mini GraphNet-style workloads on C500.

This script complements paddle_mini_graphnet_probe.py. The mini probe validates
Paddle dynamic graph and paddle.jit.save/load execution. This probe takes the
next inference-oriented step: export fixed-shape static models, load them through
paddle.inference.Config, and run PaddleInferPredictor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
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


def percentile(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


def export_model(
    name: str,
    layer: paddle.nn.Layer,
    input_specs: list[paddle.static.InputSpec],
    model_root: Path,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name}
    try:
        layer.eval()
        static_layer = paddle.jit.to_static(layer, input_spec=input_specs)
        prefix = model_root / name / "model"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paddle.jit.save(static_layer, str(prefix))
        rec.update(
            {
                "ok": True,
                "prefix": str(prefix),
                "model_file": str(prefix.with_suffix(".pdmodel")),
                "params_file": str(prefix.with_suffix(".pdiparams")),
                "files": sorted(p.name for p in prefix.parent.iterdir()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-5000:]})
    return rec


def create_predictor(model_file: str, params_file: str, use_gpu: bool, gpu_mem_mb: int) -> Any:
    from paddle import inference

    config = inference.Config(model_file, params_file)
    config.switch_ir_optim(True)
    try:
        config.enable_memory_optim()
    except Exception:
        pass
    if use_gpu:
        config.enable_use_gpu(gpu_mem_mb, 0)
    else:
        config.disable_gpu()
    return inference.create_predictor(config)


def run_predictor(
    name: str,
    export_rec: dict[str, Any],
    feeds: dict[str, np.ndarray],
    warmup: int,
    iters: int,
    use_gpu: bool,
    gpu_mem_mb: int,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name, "use_gpu": use_gpu, "warmup": warmup, "iters": iters}
    if not export_rec.get("ok"):
        rec.update({"ok": False, "error": "export_failed", "export": export_rec})
        return rec
    try:
        predictor = create_predictor(export_rec["model_file"], export_rec["params_file"], use_gpu, gpu_mem_mb)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()

        def once() -> list[np.ndarray]:
            for input_name in input_names:
                handle = predictor.get_input_handle(input_name)
                arr = feeds[input_name]
                handle.reshape(arr.shape)
                handle.copy_from_cpu(arr)
            predictor.run()
            return [predictor.get_output_handle(n).copy_to_cpu() for n in output_names]

        outs = []
        for _ in range(warmup):
            outs = once()
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            outs = once()
            sync()
            times.append((time.perf_counter() - t0) * 1000)
        arr = sorted(times)
        first = outs[0]
        rec.update(
            {
                "ok": True,
                "input_names": input_names,
                "output_names": output_names,
                "avg_ms": round(sum(times) / len(times), 4),
                "min_ms": round(arr[0], 4),
                "p50_ms": round(percentile(arr, 0.5), 4),
                "p90_ms": round(percentile(arr, 0.9), 4),
                "max_ms": round(arr[-1], 4),
                "output_shape": list(first.shape),
                "output_dtype": str(first.dtype),
                "output_sum": float(first.astype("float32").sum()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-6000:]})
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
    parser.add_argument("--gpu-mem-mb", type=int, default=2048)
    parser.add_argument("--keep-model-dir", default="")
    parser.add_argument("--sample-mxsmi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model_root = Path(tempfile.mkdtemp(prefix="paddle_infer_graphnet_"))
    model_root = Path(args.keep_model_dir) if args.keep_model_dir else temp_model_root
    rec: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": vars(args),
        "python": sys.version,
        "executable": sys.executable,
        "paddle_version": paddle.__version__,
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("PADDLE", "FLAGS", "MACA", "LD_"))},
    }
    try:
        paddle.device.set_device(args.device)
        rec["device"] = paddle.device.get_device()
        if args.sample_mxsmi:
            rec["mxsmi_before"] = run_cmd(["mx-smi"], timeout=10)

        x = paddle.randn([args.nodes, args.hidden], dtype="float32")
        src = paddle.randint(0, args.nodes, [args.edges], dtype="int64")
        dst = paddle.randint(0, args.nodes, [args.edges], dtype="int64")
        x_np = x.numpy().astype("float32")
        src_np = src.numpy().astype("int64")
        dst_np = dst.numpy().astype("int64")

        dense = DenseBlock(args.hidden)
        mixed = MixedGraphBlock(args.nodes, args.hidden)
        dense.eval()
        mixed.eval()

        exports = {
            "dense_block": export_model(
                "dense_block",
                dense,
                [paddle.static.InputSpec(shape=[args.nodes, args.hidden], dtype="float32", name="x")],
                model_root,
            ),
            "mixed_graph_block": export_model(
                "mixed_graph_block",
                mixed,
                [
                    paddle.static.InputSpec(shape=[args.nodes, args.hidden], dtype="float32", name="x"),
                    paddle.static.InputSpec(shape=[args.edges], dtype="int64", name="src"),
                    paddle.static.InputSpec(shape=[args.edges], dtype="int64", name="dst"),
                ],
                model_root,
            ),
        }
        rec["exports"] = exports
        rec["predictor_results"] = {
            "dense_block_gpu": run_predictor(
                "dense_block",
                exports["dense_block"],
                {"x": x_np},
                args.warmup,
                args.iters,
                True,
                args.gpu_mem_mb,
            ),
            "mixed_graph_block_gpu": run_predictor(
                "mixed_graph_block",
                exports["mixed_graph_block"],
                {"x": x_np, "src": src_np, "dst": dst_np},
                args.warmup,
                args.iters,
                True,
                args.gpu_mem_mb,
            ),
        }
        if args.sample_mxsmi:
            rec["mxsmi_after"] = run_cmd(["mx-smi"], timeout=10)
        rec["ok"] = all(v.get("ok") for v in rec["predictor_results"].values())
        rec["model_root"] = str(model_root) if args.keep_model_dir else ""
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()[-8000:]
    finally:
        if not args.keep_model_dir:
            shutil.rmtree(temp_model_root, ignore_errors=True)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({"ok": rec.get("ok"), "predictor_results": rec.get("predictor_results")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
