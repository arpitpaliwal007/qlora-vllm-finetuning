"""Endpoint evaluator, streaming benchmark and API smoke test against a MOCK OpenAI-compatible server.

The mock echoes the first reference answer so the scoring/identity plumbing can be
checked on a CPU. This validates request/response handling only; it is not
evidence about vLLM, the quantized model, or deployment performance.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from qlora_vllm import bench, evaluate
from qlora_vllm.artifacts import write_artifact_manifest
from qlora_vllm.prepare_data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def make_server(served_id: str, answers: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # silence
            pass

        def do_GET(self):
            if self.path == "/health":
                body = b"OK"
            elif self.path == "/v1/models":
                body = json.dumps({"data": [{"id": served_id}]}).encode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["model"] == served_id and payload["temperature"] == 0.0
            user = payload["messages"][-1]["content"]
            answer = next((a for q, a in answers.items() if q in user), "1889")
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in answer.split(" "):
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece + ' '}}]})}\n\n".encode())
                usage = {"prompt_tokens": 50, "completion_tokens": len(answer.split())}
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            body = json.dumps({"choices": [{"message": {"content": answer}}], "usage": {"prompt_tokens": 50, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def mock_endpoint(tmp_path, prepared_dataset):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "config.json").write_text("{}")
    (candidate / "model.safetensors").write_bytes(b"mock")
    manifest = write_artifact_manifest(candidate, kind="quantized_w4a16_compressed_tensors", base_model="mock", parent_artifact_id="p" * 64)
    records = read_jsonl(prepared_dataset / "test.jsonl")
    answers = {r["question"]: r["answers"][0] for r in records}
    server = make_server(manifest["artifact_id"], answers)
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield {"base_url": base_url, "candidate": candidate, "artifact_id": manifest["artifact_id"], "test": prepared_dataset / "test.jsonl"}
    server.shutdown()


def test_openai_backend_scores_and_verifies_identity(mock_endpoint, tmp_path):
    out = tmp_path / "quantized"
    evaluate.main(
        [
            "run",
            "--backend",
            "openai",
            "--base-url",
            mock_endpoint["base_url"],
            "--candidate-dir",
            str(mock_endpoint["candidate"]),
            "--test",
            str(mock_endpoint["test"]),
            "--out",
            str(out),
            "--variant",
            "quantized",
        ]
    )
    results = json.loads((out / "results.json").read_text())
    assert results["model"]["artifact_id"] == mock_endpoint["artifact_id"] and results["model"]["identity_verified"]
    assert results["model"]["served_model_id"] == mock_endpoint["artifact_id"]
    assert results["metrics"]["exact_match"] == 1.0 and results["metrics"]["f1"] == 1.0 and results["metrics"]["error_rate"] == 0.0
    assert results["timing"]["model_load_seconds"] is None and "unavailable" in results["timing"]["model_load_note"]

    # a candidate dir that does not match the served id is recorded as unverified (the gate then fails closed)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}")
    (other / "model.safetensors").write_bytes(b"different")
    write_artifact_manifest(other, kind="quantized_w4a16_compressed_tensors")
    evaluate.main(
        [
            "run",
            "--backend",
            "openai",
            "--base-url",
            mock_endpoint["base_url"],
            "--candidate-dir",
            str(other),
            "--test",
            str(mock_endpoint["test"]),
            "--out",
            str(tmp_path / "q2"),
        ]
    )
    assert json.loads((tmp_path / "q2" / "results.json").read_text())["model"]["identity_verified"] is False


def test_bench_parses_streaming_and_reports_unavailable_measurements(mock_endpoint, tmp_path):
    out = tmp_path / "bench.json"
    bench.main(
        [
            "--base-url",
            mock_endpoint["base_url"],
            "--candidate-dir",
            str(mock_endpoint["candidate"]),
            "--prompts",
            str(mock_endpoint["test"]),
            "--requests",
            "6",
            "--concurrency",
            "2",
            "--warmup",
            "2",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert report["model"]["identity_verified"] and report["settings"] == {
        "requests": 6,
        "concurrency": 2,
        "warmup_requests": 2,
        "max_tokens": 48,
        "prompt_source": str(mock_endpoint["test"]),
        "distinct_prompts": 6,
        "decoding": "greedy, streaming",
    }
    r = report["results"]
    assert r["completed"] == 6 and r["failed"] == 0 and r["token_count_source"] == ["usage"]
    assert r["request_latency_s"]["p95"] > 0 and r["time_to_first_token_s"]["p50"] > 0 and r["output_tokens_per_second"] > 0
    assert report["model_load_seconds"] is None and "unavailable" in report["model_load_note"]
    if report["hardware"]["nvidia_smi"] is None:
        assert "unavailable" in report["gpu_memory"]["note"]


def test_smoke_test_script(mock_endpoint, tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_test_api.py", "--base-url", mock_endpoint["base_url"], "--candidate-dir", str(mock_endpoint["candidate"])],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and "[smoke] OK" in proc.stdout, proc.stdout + proc.stderr
    other = tmp_path / "wrong"
    other.mkdir()
    (other / "config.json").write_text("{}")
    (other / "model.safetensors").write_bytes(b"x")
    write_artifact_manifest(other, kind="quantized_w4a16_compressed_tensors")
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_test_api.py", "--base-url", mock_endpoint["base_url"], "--candidate-dir", str(other)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "is not served" in proc.stdout
