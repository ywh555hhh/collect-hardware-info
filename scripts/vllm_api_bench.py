#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM server with simple concurrent requests.

This is a pragmatic smoke/steady-state harness, not a production benchmark.
It keeps one vLLM API server alive, sends repeated chat completion requests,
and records latency, token usage, and server lifecycle logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, body: dict[str, Any], timeout: int) -> tuple[int, str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local server only
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, repr(exc)


def get_text(url: str, timeout: int = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local server only
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, repr(exc)


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    ordered = sorted(values)
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return round(ordered[idx], 3)
    return {
        "count": len(values),
        "min": round(ordered[0], 3),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.mean(values), 3),
    }


def request_once(args: argparse.Namespace, idx: int, concurrency: int) -> dict[str, Any]:
    body = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    status, text = post_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, args.request_timeout_s)
    dt_ms = (time.perf_counter() - t0) * 1000
    rec: dict[str, Any] = {"idx": idx, "concurrency": concurrency, "status": status, "latency_ms": round(dt_ms, 3)}
    try:
        parsed = json.loads(text)
        rec["usage"] = parsed.get("usage")
        rec["content"] = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")[:300]
        if rec["usage"]:
            completion = rec["usage"].get("completion_tokens") or 0
            rec["completion_tokens_per_s"] = round(completion / (dt_ms / 1000), 3) if dt_ms else None
    except Exception as exc:  # noqa: BLE001
        rec["parse_error"] = repr(exc)
        rec["response_head"] = text[:500]
    return rec


def run_group(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    requests = args.requests_per_concurrency
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_once, args, i, concurrency) for i in range(requests)]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    wall_s = time.perf_counter() - start
    ok = [r for r in results if r.get("status") == 200]
    latencies = [float(r["latency_ms"]) for r in ok]
    completion_tokens = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok)
    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "ok": len(ok),
        "failed": requests - len(ok),
        "wall_s": round(wall_s, 3),
        "latency_ms": summarize_numbers(latencies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s_wall": round(completion_tokens / wall_s, 3) if wall_s else None,
        "results": sorted(results, key=lambda x: x["idx"]),
    }


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str]:
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
    log_file = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    proc._codex_log_file = log_file  # type: ignore[attr-defined]
    return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests-per-concurrency", type=int, default=4)
    parser.add_argument("--prompt", default="用两句话解释 KV cache 对推理服务并发的影响。")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--ready-timeout-s", type=int, default=300)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".server.log")
    proc = start_server(args, log_path)
    record: dict[str, Any] = {
        "model": args.model_name,
        "model_path": args.model,
        "params": vars(args),
        "server_pid": proc.pid,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server_log": str(log_path),
    }
    try:
        t0 = time.perf_counter()
        ready = False
        for _ in range(args.ready_timeout_s):
            status, body = get_text(f"http://127.0.0.1:{args.port}/v1/models")
            if status == 200:
                ready = True
                record["models_response"] = body
                break
            if proc.poll() is not None:
                record["server_exited_early"] = proc.returncode
                break
            time.sleep(1)
        record["ready"] = ready
        record["startup_s"] = round(time.perf_counter() - t0, 3)
        if ready:
            groups = []
            for c in [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]:
                groups.append(run_group(args, c))
            record["groups"] = groups
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        log_file = getattr(proc, "_codex_log_file", None)
        if log_file:
            log_file.close()
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["server_log_tail"] = log_path.read_text(errors="ignore")[-12000:] if log_path.exists() else ""
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({
        "model": record.get("model"),
        "ready": record.get("ready"),
        "startup_s": record.get("startup_s"),
        "groups": [
            {k: g.get(k) for k in ["concurrency", "ok", "failed", "wall_s", "completion_tokens_per_s_wall", "latency_ms"]}
            for g in record.get("groups", [])
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
