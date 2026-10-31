"""Release gate and promotion behaviour.

All reports/artifacts here are SYNTHETIC GATE-TEST FIXTURES generated in the
test; the numbers are invented to exercise the policy logic and are not model
results of any kind.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from qlora_vllm import promote, release_gate
from qlora_vllm.artifacts import read_artifact_manifest, write_artifact_manifest

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).parent / "fixtures" / "example_release_policy.yaml"  # illustrative thresholds the synthetic fixtures target
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
TEST_SHA = "a" * 64
CFG_HASH = "c" * 64


def make_artifact(path: Path, kind: str, parent: str | None = None, created: datetime = NOW - timedelta(hours=2)) -> dict:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"synthetic": "GATE TEST FIXTURE", "kind": kind}))
    (path / "model.safetensors").write_bytes(f"fake-weights-{kind}-{path.name}".encode())
    manifest = write_artifact_manifest(path, kind=kind, base_model="fixture", parent_artifact_id=parent, dataset_hash="d" * 64)
    manifest["created_at"] = created.isoformat(timespec="seconds")
    (path / "artifact_manifest.json").write_text(json.dumps(manifest))
    return manifest


def make_report(
    path: Path,
    artifact_id: str,
    f1: float,
    em: float,
    empty: float = 0.0,
    error: float = 0.0,
    created: datetime = NOW - timedelta(hours=1),
    verified: bool = True,
    test_sha: str = TEST_SHA,
    cfg_hash: str = CFG_HASH,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    report = {
        "_synthetic": "GATE TEST FIXTURE - invented numbers, not measured results",
        "schema_version": 1,
        "created_at": created.isoformat(timespec="seconds"),
        "variant": "fixture",
        "model": {"artifact_id": artifact_id, "identity_verified": verified, "backend": "fixture"},
        "dataset": {"sha256": test_sha, "dataset_hash": "d" * 64, "n": 100},
        "eval_config": {"max_new_tokens": 48},
        "eval_config_hash": cfg_hash,
        "metrics": {"exact_match": em, "f1": f1, "empty_rate": empty, "error_rate": error, "n": 100, "n_empty": int(empty * 100), "n_errors": int(error * 100)},
        "predictions_path": "predictions.jsonl",
    }
    (path / "results.json").write_text(json.dumps(report))
    (path / "predictions.jsonl").write_text("")
    return path / "results.json"


def make_bench(
    path: Path,
    artifact_id: str,
    p95: float = 1.0,
    ttft: float = 0.2,
    tps: float = 120.0,
    completed: int = 64,
    errors: float = 0.0,
    created: datetime = NOW - timedelta(hours=1),
    verified: bool = True,
) -> Path:
    bench = {
        "_synthetic": "GATE TEST FIXTURE",
        "created_at": created.isoformat(timespec="seconds"),
        "model": {"artifact_id": artifact_id, "served_model_id": artifact_id, "identity_verified": verified},
        "results": {
            "completed": completed,
            "error_rate": errors,
            "output_tokens_per_second": tps,
            "request_latency_s": {"p95": p95},
            "time_to_first_token_s": {"p95": ttft},
        },
    }
    path.write_text(json.dumps(bench))
    return path


@pytest.fixture
def world(tmp_path: Path):
    """A consistent passing candidate: merged -> quantized lineage, fresh comparable reports, good bench."""
    merged = make_artifact(tmp_path / "merged", "merged_model")
    quant = make_artifact(tmp_path / "quant", "quantized_w4a16_compressed_tensors", parent=merged["artifact_id"])
    paths = {
        "candidate_dir": tmp_path / "quant",
        "candidate": make_report(tmp_path / "r_quant", quant["artifact_id"], f1=0.80, em=0.70),
        "finetuned": make_report(tmp_path / "r_ft", merged["artifact_id"], f1=0.81, em=0.71),
        "baseline": make_report(tmp_path / "r_base", "b" * 64, f1=0.79, em=0.69),
        "bench": make_bench(tmp_path / "bench.json", quant["artifact_id"]),
        "out": tmp_path / "decision.json",
        "ids": {"merged": merged["artifact_id"], "quant": quant["artifact_id"]},
        "tmp": tmp_path,
    }
    return paths


def gate(w, policy=POLICY, **overrides) -> dict:
    args = {
        "candidate": w["candidate"],
        "candidate_dir": w["candidate_dir"],
        "finetuned": w["finetuned"],
        "baseline": w["baseline"],
        "bench": w["bench"],
        "out": w["out"],
    }
    args.update(overrides)
    argv = ["--policy", str(policy), "--now", NOW.isoformat(), "--verify-weights"]
    for key, value in args.items():
        if value is not None:
            argv += [f"--{key.replace('_', '-')}", str(value)]
    code = release_gate.main(argv)
    decision = json.loads(Path(args["out"]).read_text())
    assert (code == 0) == (decision["decision"] == "PASS")
    return decision


def failed(decision: dict) -> set[str]:
    return set(decision["summary"]["failed_checks"])


def test_passing_candidate_is_accepted(world):
    d = gate(world)
    assert d["decision"] == "PASS" and not failed(d)
    assert d["candidate"]["artifact_id"] == world["ids"]["quant"]
    assert d["candidate"]["parent_artifact_id"] == world["ids"]["merged"]
    assert d["references"]["finetuned"]["artifact_id"] == world["ids"]["merged"]
    assert d["dataset"]["test_sha256"] == TEST_SHA and d["eval_config_hash"] == CFG_HASH
    names = {c["name"] for c in d["checks"]}
    assert {
        "min_f1",
        "f1_regression_vs_baseline",
        "f1_quantization_degradation",
        "max_empty_rate",
        "max_error_rate",
        "bench_p95_latency_seconds",
        "candidate_lineage",
    } <= names
    check = next(c for c in d["checks"] if c["name"] == "min_f1")
    assert check["observed"] == 0.80 and check["threshold"] == 0.70 and check["status"] == "PASS"


def test_quality_floor_and_regressions_are_rejected(world):
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.60, em=0.50)
    assert {"min_f1", "min_exact_match", "f1_regression_vs_baseline", "f1_quantization_degradation"} <= failed(gate(world))
    # regression vs baseline only
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70)
    make_report(world["tmp"] / "r_base", "b" * 64, f1=0.90, em=0.80)
    assert failed(gate(world)) == {"f1_regression_vs_baseline", "exact_match_regression_vs_baseline"}
    # quantization degradation only
    make_report(world["tmp"] / "r_base", "b" * 64, f1=0.79, em=0.69)
    make_report(world["tmp"] / "r_ft", world["ids"]["merged"], f1=0.86, em=0.76)
    assert failed(gate(world)) == {"f1_quantization_degradation", "exact_match_quantization_degradation"}


def test_empty_and_error_rates_are_rejected(world):
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70, empty=0.05, error=0.02)
    assert failed(gate(world)) == {"max_empty_rate", "max_error_rate"}


def test_missing_or_malformed_evidence_fails_closed(world):
    assert "finetuned_report_valid" in failed(gate(world, finetuned=world["tmp"] / "nope.json"))
    assert "bench_report_valid" in failed(gate(world, bench=None))
    world["candidate"].write_text("{not json")
    d = gate(world)
    assert d["decision"] == "FAIL" and "candidate_report_valid" in failed(d)
    world["candidate"].write_text(json.dumps({"metrics": {"f1": 0.9}}))
    assert "candidate_report_valid" in failed(gate(world))


def test_stale_reports_are_rejected(world):
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70, created=NOW - timedelta(hours=100))
    assert "candidate_report_fresh" in failed(gate(world))
    # a report older than the artifact it claims to measure is also stale
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70, created=NOW - timedelta(hours=3))
    assert "candidate_report_fresh" in failed(gate(world))


def test_mismatched_artifacts_are_rejected(world):
    # report measured some other artifact
    make_report(world["tmp"] / "r_quant", "f" * 64, f1=0.80, em=0.70)
    assert "candidate_artifact_identity" in failed(gate(world))
    # endpoint identity not verified
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70, verified=False)
    assert "candidate_artifact_identity" in failed(gate(world))
    # fine-tuned report is for a model the candidate was not quantized from
    make_report(world["tmp"] / "r_quant", world["ids"]["quant"], f1=0.80, em=0.70)
    make_report(world["tmp"] / "r_ft", "e" * 64, f1=0.81, em=0.71)
    assert "candidate_lineage" in failed(gate(world))
    # different test set / eval config
    make_report(world["tmp"] / "r_ft", world["ids"]["merged"], f1=0.81, em=0.71, test_sha="9" * 64)
    assert "finetuned_report_comparable" in failed(gate(world))
    make_report(world["tmp"] / "r_ft", world["ids"]["merged"], f1=0.81, em=0.71, cfg_hash="9" * 64)
    assert "finetuned_report_comparable" in failed(gate(world))
    # benchmark for a different artifact
    make_report(world["tmp"] / "r_ft", world["ids"]["merged"], f1=0.81, em=0.71)
    make_bench(world["bench"], "f" * 64)
    assert "bench_report_valid" in failed(gate(world))
    # weights on disk changed after the manifest was written
    make_bench(world["bench"], world["ids"]["quant"])
    (world["candidate_dir"] / "model.safetensors").write_bytes(b"tampered")
    assert "candidate_artifact_identity" in failed(gate(world))


def test_performance_limits(world, tmp_path):
    make_bench(world["bench"], world["ids"]["quant"], p95=5.0, tps=10.0, completed=8)
    assert failed(gate(world)) == {"bench_p95_latency_seconds", "bench_output_tokens_per_second", "bench_min_requests"}
    policy = yaml.safe_load(POLICY.read_text())
    policy["performance"]["required"] = False
    relaxed = tmp_path / "relaxed.yaml"
    relaxed.write_text(yaml.safe_dump(policy))
    d = gate(world, policy=relaxed, bench=None)
    assert d["decision"] == "PASS" and next(c for c in d["checks"] if c["name"] == "bench_report_valid")["status"] == "SKIP"


def test_missing_baseline_depends_on_policy(world, tmp_path):
    d = gate(world, baseline=None)
    assert d["decision"] == "PASS" and next(c for c in d["checks"] if c["name"] == "baseline_report_valid")["status"] == "SKIP"
    policy = yaml.safe_load(POLICY.read_text())
    policy["evidence"]["allow_missing_baseline"] = False
    strict = tmp_path / "strict.yaml"
    strict.write_text(yaml.safe_dump(policy))
    assert "baseline_report_valid" in failed(gate(world, policy=strict, baseline=None))


def test_promotion_requires_pass_and_matching_artifact_and_keeps_rollback(world, tmp_path, monkeypatch):
    approved = tmp_path / "approved"
    decision = gate(world)
    assert decision["decision"] == "PASS"
    # freshness is checked against the real clock; pin created_at to now for the promotion step
    decision["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    world["out"].write_text(json.dumps(decision))
    promote.main(["--decision", str(world["out"]), "--candidate-dir", str(world["candidate_dir"]), "--approved-dir", str(approved)])
    current = approved / "current"
    assert read_artifact_manifest(current)["artifact_id"] == world["ids"]["quant"]
    assert (current / "release" / "release_decision.json").exists() and (current / "release" / "results.json").exists()
    history = json.loads((approved / "history.json").read_text())
    assert history[-1]["action"] == "promote" and history[-1]["previous"] is None

    # a second candidate: promotion moves the first into previous-*, rollback restores it
    merged2 = make_artifact(tmp_path / "merged2", "merged_model")
    quant2 = make_artifact(tmp_path / "quant2", "quantized_w4a16_compressed_tensors", parent=merged2["artifact_id"])
    d2 = dict(decision, candidate=dict(decision["candidate"], artifact_id=quant2["artifact_id"], path=str(tmp_path / "quant2")))
    (tmp_path / "d2.json").write_text(json.dumps(d2))
    promote.main(["--decision", str(tmp_path / "d2.json"), "--candidate-dir", str(tmp_path / "quant2"), "--approved-dir", str(approved)])
    assert read_artifact_manifest(current)["artifact_id"] == quant2["artifact_id"]
    assert len(list(approved.glob("previous-*"))) == 1
    promote.main(["--rollback", "--approved-dir", str(approved)])
    assert read_artifact_manifest(current)["artifact_id"] == world["ids"]["quant"]

    # refusals: FAIL decision, mismatched artifact, stale decision, tampered weights
    bad = dict(decision, decision="FAIL")
    (tmp_path / "fail.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="FAIL"):
        promote.main(["--decision", str(tmp_path / "fail.json"), "--candidate-dir", str(world["candidate_dir"]), "--approved-dir", str(approved)])
    with pytest.raises(SystemExit, match="decision is for artifact"):
        promote.main(["--decision", str(world["out"]), "--candidate-dir", str(tmp_path / "quant2"), "--approved-dir", str(approved)])
    old = dict(decision, created_at=(datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds"))
    (tmp_path / "old.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="older"):
        promote.main(["--decision", str(tmp_path / "old.json"), "--candidate-dir", str(world["candidate_dir"]), "--approved-dir", str(approved)])
    (world["candidate_dir"] / "model.safetensors").write_bytes(b"tampered after approval")
    with pytest.raises(ValueError, match="modified"):
        promote.main(["--decision", str(world["out"]), "--candidate-dir", str(world["candidate_dir"]), "--approved-dir", str(approved)])
    assert read_artifact_manifest(current)["artifact_id"] == world["ids"]["quant"]  # approved slot untouched by refusals


def test_shipped_policy_accepts_the_measured_full_run_and_blocks_regressions(tmp_path):
    """The calibrated configs/release_policy.yaml, fed the *measured* 2026-10-31 full-run numbers.

    The reports are rebuilt here from those numbers (artifact ids are fixture ids, not the real ones),
    so this is a policy-logic test, not a re-measurement.
    """
    shipped = ROOT / "configs" / "release_policy.yaml"
    merged = make_artifact(tmp_path / "merged", "merged_model")
    quant = make_artifact(tmp_path / "quant", "quantized_w4a16_compressed_tensors", parent=merged["artifact_id"])
    w = {
        "candidate_dir": tmp_path / "quant",
        "candidate": make_report(tmp_path / "r_quant", quant["artifact_id"], f1=0.913433, em=0.834),
        "finetuned": make_report(tmp_path / "r_ft", merged["artifact_id"], f1=0.919146, em=0.844),
        "baseline": make_report(tmp_path / "r_base", "b" * 64, f1=0.913433, em=0.834),  # the promoted artifact itself
        "bench": make_bench(tmp_path / "bench.json", quant["artifact_id"], p95=0.2246, ttft=0.0807, tps=254.15, completed=128),
        "out": tmp_path / "decision.json",
        "tmp": tmp_path,
    }
    assert gate(w, policy=shipped)["decision"] == "PASS"
    # first release without a baseline is no longer allowed by the shipped policy
    assert "baseline_report_valid" in failed(gate(w, policy=shipped, baseline=None))
    # a candidate that merely matches the base model (EM 0.701 / F1 0.806) is rejected by the floors
    make_report(tmp_path / "r_quant", quant["artifact_id"], f1=0.806, em=0.701)
    assert {"min_f1", "min_exact_match"} <= failed(gate(w, policy=shipped))
    # a quantization that costs 3 EM points is rejected
    make_report(tmp_path / "r_quant", quant["artifact_id"], f1=0.90, em=0.814)
    assert "exact_match_quantization_degradation" in failed(gate(w, policy=shipped))
    # serving at half the measured throughput is rejected
    make_report(tmp_path / "r_quant", quant["artifact_id"], f1=0.913433, em=0.834)
    make_bench(w["bench"], quant["artifact_id"], p95=0.6, ttft=0.0807, tps=100.0, completed=128)
    assert {"bench_p95_latency_seconds", "bench_output_tokens_per_second"} <= failed(gate(w, policy=shipped))
