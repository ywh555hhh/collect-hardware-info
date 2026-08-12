#!/usr/bin/env python3
"""Probe vLLM prefix-cache behavior with mixed prompt workloads.

The script keeps one OpenAI-compatible vLLM API server alive and compares:

- unique_prefix_control: each request starts with a different long prefix
- shared_prefix_cold: requests share a long prefix before explicit warmup
- shared_prefix_cached: same shared prefix after warmup requests populate cache

It records streaming TTFT-like latency, TPOT-like latency, token usage, wall
throughput, coarse mx-smi samples, and /metrics snapshots.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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


def get_text(url: str, timeout: int = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local server only
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, repr(exc)


def post_stream_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        "status": 0,
        "first_event_ms": None,
        "first_content_ms": None,
        "latency_ms": None,
        "event_count": 0,
        "content_chunk_count": 0,
        "content_chars": 0,
        "usage": None,
        "finish_reason": None,
        "stream_parse_errors": [],
        "content_chunk_gap_ms": [],
        "content_head": "",
    }
    last_content_ms: float | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local server only
            rec["status"] = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                now_ms = (time.perf_counter() - t0) * 1000
                if rec["first_event_ms"] is None:
                    rec["first_event_ms"] = round(now_ms, 3)
                rec["event_count"] += 1
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as exc:
                    rec["stream_parse_errors"].append({"error": str(exc), "payload_head": payload[:300]})
                    continue
                if parsed.get("usage"):
                    rec["usage"] = parsed.get("usage")
                choice = (parsed.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    rec["finish_reason"] = choice.get("finish_reason")
                content = (choice.get("delta") or {}).get("content") or ""
                if content:
                    if rec["first_content_ms"] is None:
                        rec["first_content_ms"] = round(now_ms, 3)
                    if last_content_ms is not None:
                        rec["content_chunk_gap_ms"].append(round(now_ms - last_content_ms, 3))
                    last_content_ms = now_ms
                    rec["content_chunk_count"] += 1
                    rec["content_chars"] += len(content)
                    if len(rec["content_head"]) < 240:
                        rec["content_head"] = (rec["content_head"] + content)[:240]
    except urllib.error.HTTPError as exc:
        rec["status"] = exc.code
        rec["response_head"] = exc.read().decode("utf-8", errors="replace")[:500]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
    finally:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return rec


def run_command_text(cmd: list[str], timeout: int = 5) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


class MxSmiSampler:
    def __init__(self, enabled: bool, interval_s: float) -> None:
        self.enabled = enabled
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def __enter__(self) -> "MxSmiSampler":
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            self.samples.append(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "usage": run_command_text(["mx-smi", "--show-usage"]),
                    "ap_usage": run_command_text(["mx-smi", "--show-ap-usage"]),
                    "hbm_bandwidth": run_command_text(["mx-smi", "--show-hbm-bandwidth"]),
                }
            )
            self._stop.wait(max(0.1, self.interval_s - (time.time() - started)))


def shared_prefix(prefix_repeats: int) -> str:
    line = (
        "Prefix cache experiment shared context: serving systems reuse identical prompt blocks; "
        "prefill computes prompt states; decode consumes cached state and generates tokens. "
        "Keep this paragraph byte-identical across requests."
    )
    return "\n".join(f"{i:03d}. {line}" for i in range(prefix_repeats))


def unique_prefix(prefix_repeats: int, request_id: int) -> str:
    # Change the first cache block so vLLM cannot reuse the long prefix from
    # other requests, while keeping the rest of the prompt length comparable.
    lines = shared_prefix(prefix_repeats).splitlines()
    lines[0] = f"{request_id:04d}. Unique prefix cache control context with request-specific first block."
    return "\n".join(lines)


def suffix_for(request_id: int) -> str:
    variants = [
        "Summarize the serving bottleneck in two concise bullets.",
        "Explain why TTFT and TPOT move differently under concurrency.",
        "Name one scheduler trade-off and one KV-cache trade-off.",
        "Give a short diagnosis of prefill-heavy mixed workloads.",
    ]
    return f"Request {request_id}: {variants[request_id % len(variants)]}"


def build_messages(args: argparse.Namespace, stage: str, request_id: int) -> list[dict[str, str]]:
    if stage.startswith("unique"):
        prefix = unique_prefix(args.prefix_repeats, request_id)
    else:
        prefix = shared_prefix(args.prefix_repeats)
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": suffix_for(request_id)},
    ]


def request_once(args: argparse.Namespace, stage: str, request_id: int) -> dict[str, Any]:
    body = {
        "model": args.model_name,
        "messages": build_messages(args, stage, request_id),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = post_stream_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, args.request_timeout_s)
    rec.update({"stage": stage, "request_id": request_id})
    usage = rec.get("usage") or {}
    completion = usage.get("completion_tokens") or 0
    latency_ms = rec.get("latency_ms") or 0
    first_content_ms = rec.get("first_content_ms")
    if completion and latency_ms:
        rec["completion_tokens_per_s"] = round(completion / (latency_ms / 1000), 3)
    if completion > 1 and first_content_ms is not None:
        rec["tpot_ms"] = round((latency_ms - float(first_content_ms)) / (completion - 1), 3)
    rec["chunk_gap_ms"] = summarize_numbers([float(x) for x in rec.get("content_chunk_gap_ms", [])])
    return rec


def metrics_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    status, text = get_text(f"http://127.0.0.1:{args.port}/metrics", timeout=10)
    prefix_lines = [line for line in text.splitlines() if "prefix" in line.lower() or "cache" in line.lower()]
    return {"status": status, "prefix_or_cache_lines": prefix_lines[-80:], "tail": text[-6000:]}


def run_stage(args: argparse.Namespace, stage: str, request_count: int, concurrency: int, id_offset: int) -> dict[str, Any]:
    metrics_before = metrics_snapshot(args)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with MxSmiSampler(args.sample_mxsmi, args.mxsmi_interval_s) as sampler:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request_once, args, stage, id_offset + i) for i in range(request_count)]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
    wall_s = time.perf_counter() - start
    metrics_after = metrics_snapshot(args)
    ok = [r for r in results if r.get("status") == 200]
    latencies = [float(r["latency_ms"]) for r in ok]
    first_content = [float(r["first_content_ms"]) for r in ok if r.get("first_content_ms") is not None]
    tpot = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok)
    completion_tokens = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok)
    return {
        "stage": stage,
        "requests": request_count,
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": request_count - len(ok),
        "wall_s": round(wall_s, 3),
        "latency_ms": summarize_numbers(latencies),
        "first_content_ms": summarize_numbers(first_content),
        "tpot_ms": summarize_numbers(tpot),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "avg_prompt_tokens": round(prompt_tokens / len(ok), 3) if ok else None,
        "avg_completion_tokens": round(completion_tokens / len(ok), 3) if ok else None,
        "completion_tokens_per_s_wall": round(completion_tokens / wall_s, 3) if wall_s else None,
        "total_tokens_per_s_wall": round((prompt_tokens + completion_tokens) / wall_s, 3) if wall_s else None,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "mxsmi_samples": sampler.samples,
        "results": sorted(results, key=lambda x: x["request_id"]),
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


def parse_prefix_hit_rates(log_text: str) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(r"Prefix cache hit rate: (?P<rate>[0-9.]+)%")
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"line": line[-500:], "rate_pct": float(match.group("rate"))})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefix-repeats", type=int, default=24)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--port", type=int, default=18040)
    parser.add_argument("--ready-timeout-s", type=int, default=360)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument("--sample-mxsmi", action="store_true")
    parser.add_argument("--mxsmi-interval-s", type=float, default=1.0)
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
        "server_log": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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
            stages = []
            stages.append(run_stage(args, "unique_prefix_control", args.requests, args.concurrency, 0))
            stages.append(run_stage(args, "shared_prefix_cold", args.requests, args.concurrency, 10_000))
            stages.append(run_stage(args, "shared_prefix_warmup", args.warmup_requests, 1, 20_000))
            stages.append(run_stage(args, "shared_prefix_cached", args.requests, args.concurrency, 30_000))
            record["stages"] = stages
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
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    record["server_log_tail"] = log_text[-16000:]
    record["prefix_cache_log_rates"] = parse_prefix_hit_rates(log_text)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "model": record["model"],
                "ready": record.get("ready"),
                "startup_s": record.get("startup_s"),
                "stages": [
                    {
                        k: stage.get(k)
                        for k in [
                            "stage",
                            "ok",
                            "failed",
                            "wall_s",
                            "avg_prompt_tokens",
                            "completion_tokens_per_s_wall",
                            "first_content_ms",
                            "tpot_ms",
                        ]
                    }
                    for stage in record.get("stages", [])
                ],
                "prefix_cache_log_rates_tail": record.get("prefix_cache_log_rates", [])[-5:],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
