#!/usr/bin/env python3
"""Sweep vLLM KV-cache configuration and summarize concurrency signals."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_key_events(log: str) -> dict[str, Any]:
    def first(pattern: str, cast=str):
        match = re.search(pattern, log)
        if not match:
            return None
        value = match.group(1).replace(",", "")
        try:
            return cast(value)
        except Exception:  # noqa: BLE001
            return value

    events: dict[str, Any] = {
        "resolved_architecture": first(r"Resolved architecture: (.+)"),
        "max_model_len": first(r"Using max model len ([0-9,]+)", int),
        "weight_load_s": first(r"Loading weights took ([0-9.]+) seconds", float),
        "model_memory_gib": first(r"Model loading took ([0-9.]+) GiB memory", float),
        "model_loading_s": first(r"Model loading took [0-9.]+ GiB memory and ([0-9.]+) seconds", float),
        "kv_available_gib": first(r"Available KV cache memory: ([0-9.]+) GiB", float),
        "kv_cache_tokens": first(r"GPU KV cache size: ([0-9,]+) tokens", int),
        "engine_init_s": first(r"init engine .* took ([0-9.]+) seconds", float),
        "attention_backend": first(r"Using ([A-Z_]+) attention backend"),
    }
    conc = re.search(r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x", log)
    if conc:
        events["concurrency_context_tokens"] = int(conc.group(1).replace(",", ""))
        events["max_concurrency"] = float(conc.group(2))
    events["mamba_page_alignment"] = re.findall(r".*mamba page size.*", log)
    events["kv_padding"] = re.findall(r"Add [0-9]+ padding layers.*KV cache memory", log)
    return events


def run_case(args: argparse.Namespace, max_model_len: int, gpu_mem: float, batch_size: int) -> dict[str, Any]:
    case_name = f"len{max_model_len}_mem{gpu_mem:.2f}_bs{batch_size}".replace(".", "p")
    log_path = Path(args.output_dir) / f"{case_name}.log"
    out_path = Path(args.output_dir) / f"{case_name}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("vllm_lifecycle_probe.py")),
        "--model",
        args.model,
        "--model-name",
        args.model_name,
        "--output",
        str(out_path),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(batch_size),
        "--prompt",
        args.prompt,
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    start = time.perf_counter()
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=args.case_timeout_s)
    wall_s = time.perf_counter() - start
    log = log_path.read_text(errors="ignore")
    rec: dict[str, Any] = {
        "case": case_name,
        "model": args.model_name,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem,
        "batch_size": batch_size,
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 3),
        "log_path": str(log_path),
        "output_path": str(out_path),
        "events": parse_key_events(log),
    }
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text())
            offline = records[0] if records else {}
            rec["ok"] = offline.get("ok")
            rec["load_wall_s"] = offline.get("load_wall_s")
            rec["generate_wall_s"] = offline.get("generate_wall_s")
            rec["aggregate_tok_s"] = offline.get("aggregate_tok_s")
            rec["total_new_tokens"] = offline.get("total_new_tokens")
            rec["error_type"] = offline.get("error_type")
            rec["error"] = offline.get("error")
            embedded_events = offline.get("events") or {}
            if rec["events"].get("resolved_architecture") is None and embedded_events.get("resolved_architecture"):
                rec["events"]["resolved_architecture"] = embedded_events["resolved_architecture"][0]
            if rec["events"].get("max_model_len") is None and embedded_events.get("max_model_len"):
                rec["events"]["max_model_len"] = int(str(embedded_events["max_model_len"][0]).replace(",", ""))
            if not rec["events"].get("mamba_page_alignment") and embedded_events.get("mamba_page_alignment"):
                rec["events"]["mamba_page_alignment"] = embedded_events["mamba_page_alignment"]
        except Exception as exc:  # noqa: BLE001
            rec["json_error"] = repr(exc)
    return rec


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-lens", default="1024,2048,4096")
    parser.add_argument("--gpu-memory-utils", default="0.50,0.70")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--case-timeout-s", type=int, default=420)
    parser.add_argument("--prompt", default="用一句话解释 KV cache sweep 的意义。")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for max_model_len in parse_csv_ints(args.max_model_lens):
        for gpu_mem in parse_csv_floats(args.gpu_memory_utils):
            for batch_size in parse_csv_ints(args.batch_sizes):
                rec = run_case(args, max_model_len, gpu_mem, batch_size)
                summary.append(rec)
                print(json.dumps(rec, ensure_ascii=False))
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
