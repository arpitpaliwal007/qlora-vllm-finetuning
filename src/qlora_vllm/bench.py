"""Benchmark an OpenAI-compatible endpoint: latency percentiles, TTFT, throughput, GPU memory.

    python -m qlora_vllm.bench --base-url http://localhost:8000/v1 --candidate-dir artifacts/quantized-w4a16 \
        --prompts data/squad_qa/test.jsonl --requests 64 --concurrency 4 --warmup 8 --out reports/bench.json

Measurements that cannot be taken are reported as ``null`` with a note rather than estimated:
model load time comes from ``serve_status.json`` (written by ``scripts/serve_vllm.sh``) when present,
and GPU memory from ``nvidia-smi`` when it is installed on the machine running this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_artifact_manifest, read_json, utc_now, write_json
from .evaluate import http_json
from .hardware import detect_hardware
from .prepare_data import read_jsonl
from .prompting import prompt_messages


def stream_request(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_token = None
    chunks = 0
    usage = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local endpoint
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        chunks += 1
                        if first_token is None:
                            first_token = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - t0}
    return {
        "error": None,
        "latency_s": time.perf_counter() - t0,
        "ttft_s": first_token,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": (usage or {}).get("completion_tokens", chunks),
        "token_count_source": "usage" if usage else "content_chunks",
    }


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99)} | {"mean": float(arr.mean()), "max": float(arr.max())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", help="served model id (optional when exactly one model is served)")
    parser.add_argument("--candidate-dir", help="artifact dir the endpoint is expected to serve")
    parser.add_argument("--prompts", required=True, help="prepared JSONL whose `messages` provide prompts")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--serve-status", default=None, help="serve_status.json written by serve_vllm.sh (model load time)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    served = [m["id"] for m in http_json(f"{base_url}/models")["data"]]
    model = args.model or (served[0] if len(served) == 1 else None)
    if model is None or model not in served:
        raise SystemExit(f"--model must be one of the served ids: {served}")
    identity: dict[str, Any] = {"served_model_id": model, "base_url": base_url}
    if args.candidate_dir:
        manifest = read_artifact_manifest(args.candidate_dir)
        identity.update({"artifact_id": manifest["artifact_id"], "kind": manifest["kind"], "identity_verified": manifest["artifact_id"] == model})

    records = read_jsonl(Path(args.prompts))
    prompts = [prompt_messages(r) for r in records]
    if not prompts:
        raise SystemExit("no prompts")
    pick = lambda i: prompts[i % len(prompts)]  # noqa: E731

    def call(i: int) -> dict[str, Any]:
        return stream_request(base_url, model, pick(i), args.max_tokens, args.timeout)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(call, range(args.warmup)))
    gpu_before = detect_hardware()["nvidia_smi"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, range(args.warmup, args.warmup + args.requests)))
    wall = time.perf_counter() - t0
    gpu_after = detect_hardware()["nvidia_smi"]

    ok = [r for r in results if r["error"] is None]
    out_tokens = sum(r["output_tokens"] or 0 for r in ok)
    serve_status = None
    status_path = Path(args.serve_status) if args.serve_status else (Path(args.candidate_dir) / "serve_status.json" if args.candidate_dir else None)
    if status_path and status_path.exists():
        serve_status = read_json(status_path)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": identity,
        "settings": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup,
            "max_tokens": args.max_tokens,
            "prompt_source": str(args.prompts),
            "distinct_prompts": min(len(prompts), args.requests),
            "decoding": "greedy, streaming",
        },
        "results": {
            "completed": len(ok),
            "failed": len(results) - len(ok),
            "error_rate": (len(results) - len(ok)) / len(results),
            "wall_seconds": wall,
            "requests_per_second": len(ok) / wall if wall else None,
            "output_tokens_per_second": out_tokens / wall if wall else None,
            "request_latency_s": percentiles([r["latency_s"] for r in ok]),
            "time_to_first_token_s": percentiles([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
            "prompt_tokens": percentiles([r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]),
            "output_tokens": percentiles([r["output_tokens"] for r in ok if r.get("output_tokens") is not None]),
            "token_count_source": sorted({r["token_count_source"] for r in ok}) if ok else None,
            "per_request_generation_tokens_per_second": percentiles(
                [r["output_tokens"] / (r["latency_s"] - r["ttft_s"]) for r in ok if r.get("ttft_s") and r["output_tokens"] and r["latency_s"] > r["ttft_s"]]
            ),
        },
        "model_load_seconds": (serve_status or {}).get("model_load_seconds"),
        "model_load_note": None if serve_status else "unavailable: no serve_status.json; model load time is only known to the process that started the server",
        "gpu_memory": {"before": gpu_before, "after": gpu_after, "note": None if gpu_after else "unavailable: nvidia-smi not found on the benchmarking host"},
        "hardware": detect_hardware(),
        "errors": [r["error"] for r in results if r["error"]][:10],
    }
    write_json(Path(args.out), report)
    print(
        json.dumps(
            {
                k: report["results"][k]
                for k in ("completed", "failed", "requests_per_second", "output_tokens_per_second", "request_latency_s", "time_to_first_token_s")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
