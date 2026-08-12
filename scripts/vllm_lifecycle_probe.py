#!/usr/bin/env python3
"""Run one vLLM model lifecycle probe and emit structured JSON.

The probe is intentionally conservative: it records the visible hardware,
config parsing, LLM load, KV-cache lines from logs, generation, and optional
OpenAI-compatible API server smoke results.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EVENT_PATTERNS = {
    "resolved_architecture": r"Resolved architecture: (?P<value>.+)",
    "casting_dtype": r"Casting (?P<value>torch\.[^ ]+ to torch\.[^ .]+)",
    "max_model_len": r"Using max model len (?P<value>[0-9,]+)",
    "chunked_prefill": r"Chunked prefill is enabled with max_num_batched_tokens=(?P<value>[0-9,]+)",
    "attention_backend": r"Using (?P<value>[A-Z_]+) attention backend",
    "maca_flash_attention": r"Using Maca version of flash attention",
    "weight_load_s": r"Loading weights took (?P<value>[0-9.]+) seconds",
    "model_memory": r"Model loading took (?P<memory>[0-9.]+ GiB) memory and (?P<seconds>[0-9.]+) seconds",
    "kv_available_memory": r"Available KV cache memory: (?P<value>[0-9.]+ GiB)",
    "kv_cache_tokens": r"GPU KV cache size: (?P<value>[0-9,]+) tokens",
    "max_concurrency": r"Maximum concurrency for (?P<context>[0-9,]+) tokens per request: (?P<value>[0-9.]+)x",
    "engine_init_s": r"init engine .* took (?P<value>[0-9.]+) seconds",
    "mamba_page_alignment": r"(?P<value>.*mamba page size.*)",
    "kv_padding": r"(?P<value>Add [0-9]+ padding layers.*KV cache memory)",
    "api_routes": r"Route: (?P<value>/v1/[^,]+)",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def model_config(model_path: Path) -> dict[str, Any]:
    cfg = read_json(model_path / "config.json")
    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "max_position_embeddings",
        "sliding_window",
        "rope_theta",
        "torch_dtype",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def generation_config(model_path: Path) -> dict[str, Any]:
    cfg = read_json(model_path / "generation_config.json")
    keys = ["bos_token_id", "eos_token_id", "pad_token_id", "temperature", "top_p", "top_k"]
    return {k: cfg.get(k) for k in keys if k in cfg}


def run_cmd(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return 0, out
    except subprocess.CalledProcessError as exc:
        return exc.returncode, exc.output
    except Exception as exc:  # noqa: BLE001
        return 1, repr(exc)


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    rc, mxsmi = run_cmd(["mx-smi"], timeout=15)
    snap["mx_smi_rc"] = rc
    snap["mx_smi"] = mxsmi
    try:
        import torch

        snap["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            snap["torch"].update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "visible_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "warp_size": props.warp_size,
                    "l2_cache_size": getattr(props, "L2_cache_size", None),
                }
            )
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = repr(exc)
    return snap


def parse_events(text: str) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = {}
    for name, pattern in EVENT_PATTERNS.items():
        for match in re.finditer(pattern, text):
            if "value" in match.groupdict():
                value: Any = match.group("value")
            else:
                value = match.groupdict()
            events.setdefault(name, []).append(value)
    return events


def run_offline(args: argparse.Namespace) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "phase": "offline_llm",
        "started_at": now(),
        "model": args.model_name,
        "model_path": args.model,
        "config": model_config(Path(args.model)),
        "generation_config": generation_config(Path(args.model)),
        "params": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "enforce_eager": args.enforce_eager,
        },
        "hardware_before": hardware_snapshot(),
    }
    logs = io.StringIO()
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            from vllm import LLM, SamplingParams

            load_t0 = time.perf_counter()
            llm = LLM(
                model=args.model,
                trust_remote_code=True,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
            )
            rec["load_wall_s"] = round(time.perf_counter() - load_t0, 3)
            prompts = [args.prompt for _ in range(args.batch_size)]
            params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
            gen_t0 = time.perf_counter()
            outs = llm.generate(prompts, params)
            gen_s = time.perf_counter() - gen_t0
            new_tokens = [len(o.outputs[0].token_ids) for o in outs]
            rec["generate_wall_s"] = round(gen_s, 3)
            rec["new_tokens"] = new_tokens
            rec["total_new_tokens"] = sum(new_tokens)
            rec["aggregate_tok_s"] = round(sum(new_tokens) / gen_s, 3) if gen_s else None
            rec["sample"] = outs[0].outputs[0].text.replace("\n", " ")[:500] if outs else ""
            rec["ok"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error_type"] = type(exc).__name__
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()[-8000:]
    rec["logs"] = logs.getvalue()
    rec["events"] = parse_events(rec["logs"])
    rec["hardware_after"] = hardware_snapshot()
    rec["finished_at"] = now()
    return rec


def run_api_server(args: argparse.Namespace) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "phase": "api_server",
        "started_at": now(),
        "model": args.model_name,
        "model_path": args.model,
        "port": args.port,
    }
    log_path = Path(args.output).with_suffix(".api_server.log")
    response_path = Path(args.output).with_suffix(".api_response.json")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.model_name,
        "--trust-remote-code",
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    with log_path.open("w") as log_file:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        rec["pid"] = proc.pid
        ready = False
        for _ in range(args.api_ready_timeout_s):
            rc, models = run_cmd(["curl", "-sf", f"http://127.0.0.1:{args.port}/v1/models"], timeout=5)
            if rc == 0:
                ready = True
                rec["models_response"] = models
                break
            if proc.poll() is not None:
                rec["server_exited_early"] = proc.returncode
                break
            time.sleep(1)
        rec["ready"] = ready
        rec["startup_s"] = round(time.perf_counter() - start, 3)
        if ready:
            body = json.dumps(
                {
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": args.prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                },
                ensure_ascii=False,
            )
            req_start = time.perf_counter()
            curl = subprocess.run(
                [
                    "curl",
                    "-s",
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    body,
                ],
                text=True,
                capture_output=True,
                timeout=args.api_request_timeout_s,
                check=False,
            )
            rec["request_ms"] = round((time.perf_counter() - req_start) * 1000, 1)
            rec["request_rc"] = curl.returncode
            response_path.write_text(curl.stdout)
            try:
                parsed = json.loads(curl.stdout)
                rec["usage"] = parsed.get("usage")
                rec["content"] = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")[:500]
            except Exception as exc:  # noqa: BLE001
                rec["response_parse_error"] = repr(exc)
                rec["response_head"] = curl.stdout[:500]
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    rec["log_path"] = str(log_path)
    rec["response_path"] = str(response_path)
    rec["logs_tail"] = log_path.read_text(errors="ignore")[-12000:]
    rec["events"] = parse_events(log_path.read_text(errors="ignore"))
    rec["finished_at"] = now()
    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt", default="用一句话解释 vLLM 模型生命周期测试的意义。")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--api-ready-timeout-s", type=int, default=300)
    parser.add_argument("--api-request-timeout-s", type=int, default=60)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [run_offline(args)]
    if args.api:
        records.append(run_api_server(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    for rec in records:
        print(
            json.dumps(
                {
                    "phase": rec.get("phase"),
                    "model": rec.get("model"),
                    "ok": rec.get("ok", rec.get("ready")),
                    "load_wall_s": rec.get("load_wall_s"),
                    "generate_wall_s": rec.get("generate_wall_s"),
                    "aggregate_tok_s": rec.get("aggregate_tok_s"),
                    "startup_s": rec.get("startup_s"),
                    "request_ms": rec.get("request_ms"),
                    "events": rec.get("events", {}),
                    "error_type": rec.get("error_type"),
                    "error": rec.get("error"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
