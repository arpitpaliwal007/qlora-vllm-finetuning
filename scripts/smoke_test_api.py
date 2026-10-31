#!/usr/bin/env python
"""Smoke test an OpenAI-compatible endpoint: health, served model identity, one grounded answer.

    python scripts/smoke_test_api.py --base-url http://localhost:8000/v1 --candidate-dir artifacts/quantized-w4a16

Exit code is nonzero when the endpoint is unhealthy, serves a different artifact
than expected, or returns an empty answer for a trivial context.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora_vllm.artifacts import read_artifact_manifest  # noqa: E402
from qlora_vllm.evaluate import http_json  # noqa: E402
from qlora_vllm.prompting import build_messages  # noqa: E402

CONTEXT = "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris. It is 330 metres tall."
QUESTION = "In which year was the Eiffel Tower completed?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    root = base.rsplit("/v1", 1)[0]
    with urllib.request.urlopen(f"{root}/health", timeout=args.timeout) as r:  # noqa: S310
        print(f"[smoke] /health -> {r.status}")
    served = [m["id"] for m in http_json(f"{base}/models")["data"]]
    print(f"[smoke] served models: {served}")
    if args.candidate_dir:
        expected = read_artifact_manifest(args.candidate_dir)["artifact_id"]
        if expected not in served:
            print(f"[smoke] FAIL: expected artifact {expected[:12]} is not served")
            return 1
        model = expected
    else:
        model = served[0]
    resp = http_json(
        f"{base}/chat/completions", {"model": model, "messages": build_messages(CONTEXT, QUESTION), "temperature": 0.0, "max_tokens": 32}, timeout=args.timeout
    )
    answer = (resp["choices"][0]["message"]["content"] or "").strip()
    print(f"[smoke] answer: {answer!r}  usage: {resp.get('usage')}")
    if not answer:
        print("[smoke] FAIL: empty answer")
        return 1
    print("[smoke] OK" + ("" if "1889" in answer else " (note: answer does not contain the expected span '1889')"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
