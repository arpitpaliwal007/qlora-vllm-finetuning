"""Release gate: decide PASS/FAIL for a candidate from evaluation evidence and a policy.

The gate fails closed: a missing, malformed, stale or mismatched report is a
FAIL with a reason, never a skipped check. Exit code 0 only on PASS.

    python -m qlora_vllm.release_gate --policy configs/release_policy.yaml \
        --candidate reports/quantized/results.json --candidate-dir artifacts/quantized-w4a16 \
        --finetuned reports/finetuned/results.json --bench reports/bench.json \
        [--baseline deploy/approved/current/release/results.json] --out reports/release_decision.json
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .artifacts import hash_artifact_dir, read_artifact_manifest, read_json, sha256_json, utc_now, write_json

DECISION_SCHEMA = 1


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "PASS"
        self.observed: Any = None
        self.threshold: Any = None
        self.reason = ""

    def fail(self, reason: str, observed: Any = None, threshold: Any = None) -> Check:
        self.status, self.reason, self.observed, self.threshold = "FAIL", reason, observed, threshold
        return self

    def ok(self, reason: str, observed: Any = None, threshold: Any = None) -> Check:
        self.status, self.reason, self.observed, self.threshold = "PASS", reason, observed, threshold
        return self

    def skip(self, reason: str) -> Check:
        self.status, self.reason = "SKIP", reason
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "observed": self.observed, "threshold": self.threshold, "reason": self.reason}


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def load_report(path: str | None, label: str, checks: list[Check]) -> dict[str, Any] | None:
    """Load and schema-check a results.json; a missing/malformed report is a failed check."""
    check = Check(f"{label}_report_valid")
    checks.append(check)
    if not path:
        check.fail("report not provided")
        return None
    p = Path(path)
    if not p.exists():
        check.fail(f"report not found: {p}")
        return None
    try:
        report = read_json(p)
        metrics = report["metrics"]
        for key in ("exact_match", "f1", "empty_rate", "error_rate", "n"):
            float(metrics[key])
        for key in ("artifact_id",):
            if not report["model"].get(key):
                raise KeyError(f"model.{key}")
        report["dataset"]["sha256"], report["eval_config_hash"], report["created_at"]
    except Exception as exc:  # noqa: BLE001
        check.fail(f"malformed report {p}: {type(exc).__name__}: {exc}")
        return None
    report["_path"] = str(p)
    check.ok(f"loaded {p}")
    return report


def check_fresh(report: dict[str, Any], label: str, max_age_hours: float, now: datetime, not_before: datetime | None, checks: list[Check]) -> None:
    check = Check(f"{label}_report_fresh")
    checks.append(check)
    created = parse_time(report.get("created_at"))
    if created is None:
        check.fail("report has no created_at")
        return
    age_h = (now - created).total_seconds() / 3600
    if age_h > max_age_hours:
        check.fail("report is stale", observed=round(age_h, 1), threshold=max_age_hours)
    elif age_h < -1:
        check.fail("report is dated in the future", observed=round(age_h, 1), threshold=max_age_hours)
    elif not_before and created < not_before:
        check.fail(f"report ({created.isoformat()}) predates the artifact it claims to measure ({not_before.isoformat()})")
    else:
        check.ok("report is within the freshness window", observed=round(age_h, 1), threshold=max_age_hours)


def check_comparable(candidate: dict[str, Any], other: dict[str, Any], label: str, checks: list[Check]) -> None:
    check = Check(f"{label}_report_comparable")
    checks.append(check)
    same_dataset = candidate["dataset"]["sha256"] == other["dataset"]["sha256"]
    same_config = candidate["eval_config_hash"] == other["eval_config_hash"]
    if same_dataset and same_config:
        check.ok("same test set and evaluation configuration")
    else:
        check.fail(
            "reports measure different data or settings",
            observed={"same_test_sha256": same_dataset, "same_eval_config": same_config},
            threshold="identical",
        )


def check_max(name: str, observed: float | None, threshold: float | None, checks: list[Check], what: str) -> None:
    check = Check(name)
    checks.append(check)
    if threshold is None:
        check.skip("no threshold configured")
    elif observed is None:
        check.fail(f"{what} unavailable", threshold=threshold)
    elif observed > threshold:
        check.fail(f"{what} exceeds the limit", observed=observed, threshold=threshold)
    else:
        check.ok(f"{what} within the limit", observed=observed, threshold=threshold)


def check_min(name: str, observed: float | None, threshold: float | None, checks: list[Check], what: str) -> None:
    check = Check(name)
    checks.append(check)
    if threshold is None:
        check.skip("no threshold configured")
    elif observed is None:
        check.fail(f"{what} unavailable", threshold=threshold)
    elif observed < threshold:
        check.fail(f"{what} below the minimum", observed=observed, threshold=threshold)
    else:
        check.ok(f"{what} meets the minimum", observed=observed, threshold=threshold)


def evaluate_policy(policy: dict[str, Any], args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    checks: list[Check] = []
    quality, reliability = policy.get("quality", {}), policy.get("reliability", {})
    perf, evidence = policy.get("performance", {}), policy.get("evidence", {})
    max_age = float(evidence.get("max_report_age_hours", 72))

    # ---- candidate artifact identity
    identity = Check("candidate_artifact_identity")
    checks.append(identity)
    manifest: dict[str, Any] | None = None
    try:
        manifest = read_artifact_manifest(args.candidate_dir)
        if args.verify_weights:
            current_id, _ = hash_artifact_dir(args.candidate_dir)
            if current_id != manifest["artifact_id"]:
                identity.fail("artifact files on disk no longer match their manifest", observed=current_id[:12], threshold=manifest["artifact_id"][:12])
                manifest = None
    except Exception as exc:  # noqa: BLE001
        identity.fail(f"candidate artifact manifest unusable: {type(exc).__name__}: {exc}")

    candidate = load_report(args.candidate, "candidate", checks)
    if candidate and manifest and identity.status != "FAIL":
        if candidate["model"]["artifact_id"] != manifest["artifact_id"]:
            identity.fail(
                "candidate report measured a different artifact", observed=candidate["model"]["artifact_id"][:12], threshold=manifest["artifact_id"][:12]
            )
        elif evidence.get("require_identity_verified", True) and not candidate["model"].get("identity_verified", False):
            identity.fail("candidate report could not verify which artifact the endpoint served (identity_verified=false)")
        else:
            identity.ok("report artifact id matches the candidate manifest", observed=manifest["artifact_id"][:12])
    if candidate:
        check_fresh(candidate, "candidate", max_age, now, parse_time(manifest.get("created_at")) if manifest else None, checks)

    # ---- fine-tuned (unquantized) reference and lineage
    finetuned = load_report(args.finetuned, "finetuned", checks)
    if candidate and finetuned:
        check_comparable(candidate, finetuned, "finetuned", checks)
        check_fresh(finetuned, "finetuned", max_age, now, None, checks)
        lineage = Check("candidate_lineage")
        checks.append(lineage)
        parent = (manifest or {}).get("parent_artifact_id")
        if parent is None:
            lineage.fail("candidate manifest records no parent artifact; cannot prove it was quantized from the evaluated fine-tuned model")
        elif parent != finetuned["model"]["artifact_id"]:
            lineage.fail(
                "fine-tuned report measured a different model than the candidate was quantized from",
                observed=finetuned["model"]["artifact_id"][:12],
                threshold=parent[:12],
            )
        else:
            lineage.ok("candidate was quantized from the evaluated fine-tuned model", observed=parent[:12])

    # ---- approved baseline
    baseline = None
    if args.baseline:
        baseline = load_report(args.baseline, "baseline", checks)
        if candidate and baseline:
            check_comparable(candidate, baseline, "baseline", checks)
    else:
        check = Check("baseline_report_valid")
        checks.append(check)
        if evidence.get("allow_missing_baseline", False):
            check.skip("no approved baseline supplied and policy allows a missing baseline (first release)")
        else:
            check.fail("no approved baseline supplied and policy requires one")

    # ---- quality & reliability
    cm = candidate["metrics"] if candidate else {}
    check_min("min_f1", cm.get("f1"), quality.get("min_f1"), checks, "candidate F1")
    check_min("min_exact_match", cm.get("exact_match"), quality.get("min_exact_match"), checks, "candidate exact match")
    if baseline and candidate:
        bm = baseline["metrics"]
        check_max(
            "f1_regression_vs_baseline",
            round(bm["f1"] - cm["f1"], 6),
            quality.get("max_f1_regression_vs_baseline"),
            checks,
            "F1 regression vs approved baseline",
        )
        check_max(
            "exact_match_regression_vs_baseline",
            round(bm["exact_match"] - cm["exact_match"], 6),
            quality.get("max_exact_match_regression_vs_baseline"),
            checks,
            "EM regression vs approved baseline",
        )
    elif args.baseline:
        checks.append(Check("f1_regression_vs_baseline").fail("baseline evidence unusable"))
    else:
        checks.append(Check("f1_regression_vs_baseline").skip("no baseline"))
    if finetuned and candidate:
        fm = finetuned["metrics"]
        check_max(
            "f1_quantization_degradation",
            round(fm["f1"] - cm["f1"], 6),
            quality.get("max_f1_quantization_degradation"),
            checks,
            "F1 degradation vs fine-tuned model",
        )
        check_max(
            "exact_match_quantization_degradation",
            round(fm["exact_match"] - cm["exact_match"], 6),
            quality.get("max_exact_match_quantization_degradation"),
            checks,
            "EM degradation vs fine-tuned model",
        )
    else:
        checks.append(Check("f1_quantization_degradation").fail("fine-tuned evidence unusable"))
    check_max("max_empty_rate", cm.get("empty_rate"), reliability.get("max_empty_rate"), checks, "empty-response rate")
    check_max("max_error_rate", cm.get("error_rate"), reliability.get("max_error_rate"), checks, "inference error rate")

    # ---- deployment performance
    bench = None
    if perf.get("required", False):
        bcheck = Check("bench_report_valid")
        checks.append(bcheck)
        try:
            if not args.bench:
                raise FileNotFoundError("benchmark report not provided")
            bench = read_json(args.bench)
            results = bench["results"]
            served = bench["model"]
            if manifest and served.get("artifact_id") != manifest["artifact_id"]:
                raise ValueError("benchmark measured a different artifact than the candidate")
            if evidence.get("require_identity_verified", True) and not served.get("identity_verified", False):
                raise ValueError("benchmark could not verify which artifact the endpoint served")
            bcheck.ok(f"loaded {args.bench}")
            check_fresh(bench, "bench", max_age, now, parse_time(manifest.get("created_at")) if manifest else None, checks)
            check_min("bench_min_requests", results.get("completed"), perf.get("min_requests"), checks, "completed benchmark requests")
            check_max(
                "bench_p95_latency_seconds",
                (results.get("request_latency_s") or {}).get("p95"),
                perf.get("max_p95_latency_seconds"),
                checks,
                "p95 request latency (s)",
            )
            check_max(
                "bench_p95_ttft_seconds",
                (results.get("time_to_first_token_s") or {}).get("p95"),
                perf.get("max_p95_ttft_seconds"),
                checks,
                "p95 time to first token (s)",
            )
            check_min(
                "bench_output_tokens_per_second", results.get("output_tokens_per_second"), perf.get("min_output_tokens_per_second"), checks, "output tokens/s"
            )
            check_max("bench_error_rate", results.get("error_rate"), perf.get("max_error_rate"), checks, "benchmark error rate")
        except Exception as exc:  # noqa: BLE001
            bcheck.fail(f"benchmark evidence unusable: {type(exc).__name__}: {exc}")
    else:
        checks.append(Check("bench_report_valid").skip("performance checks not required by policy"))

    failed = [c for c in checks if c.status == "FAIL"]
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": "FAIL" if failed else "PASS",
        "created_at": utc_now(),
        "evaluated_at": now.isoformat(timespec="seconds"),
        "policy": {"name": policy.get("policy_name"), "path": str(args.policy), "sha256": sha256_json(policy)},
        "candidate": {
            "path": str(args.candidate_dir),
            "artifact_id": (manifest or {}).get("artifact_id"),
            "kind": (manifest or {}).get("kind"),
            "base_model": (manifest or {}).get("base_model"),
            "parent_artifact_id": (manifest or {}).get("parent_artifact_id"),
            "report": args.candidate,
        },
        "dataset": {
            "test_sha256": candidate["dataset"]["sha256"] if candidate else None,
            "dataset_hash": candidate["dataset"].get("dataset_hash") if candidate else None,
        },
        "eval_config_hash": candidate["eval_config_hash"] if candidate else None,
        "eval_config": candidate.get("eval_config") if candidate else None,
        "references": {
            "finetuned": {"report": args.finetuned, "artifact_id": finetuned["model"]["artifact_id"] if finetuned else None},
            "baseline": {"report": args.baseline, "artifact_id": baseline["model"]["artifact_id"] if baseline else None},
            "bench": {"report": args.bench, "served_model_id": (bench or {}).get("model", {}).get("served_model_id")},
        },
        "checks": [c.as_dict() for c in checks],
        "summary": {
            "passed": sum(c.status == "PASS" for c in checks),
            "failed": len(failed),
            "skipped": sum(c.status == "SKIP" for c in checks),
            "failed_checks": [c.name for c in failed],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--candidate", required=True, help="results.json for the candidate (quantized, served)")
    parser.add_argument("--candidate-dir", required=True, help="artifact directory of the candidate")
    parser.add_argument("--finetuned", required=True, help="results.json for the unquantized fine-tuned model")
    parser.add_argument("--baseline", help="results.json of the currently approved artifact")
    parser.add_argument("--bench", help="bench.json for the served candidate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--now", help="ISO timestamp used for freshness checks (default: current UTC time)")
    parser.add_argument("--verify-weights", action="store_true", help="re-hash the candidate directory before trusting its manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = parse_time(args.now) or datetime.now(UTC)
    try:
        policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {}
        decision = evaluate_policy(policy, args, now)
    except Exception as exc:  # noqa: BLE001 - fail closed on any unexpected error
        decision = {
            "schema_version": DECISION_SCHEMA,
            "decision": "FAIL",
            "created_at": utc_now(),
            "policy": {"path": str(args.policy)},
            "candidate": {"path": str(args.candidate_dir), "report": args.candidate},
            "checks": [{"name": "gate_execution", "status": "FAIL", "observed": None, "threshold": None, "reason": f"{type(exc).__name__}: {exc}"}],
            "summary": {"passed": 0, "failed": 1, "skipped": 0, "failed_checks": ["gate_execution"]},
            "traceback": traceback.format_exc(),
        }
    write_json(Path(args.out), decision)
    for check in decision["checks"]:
        detail = "" if check["observed"] is None else f" observed={check['observed']} threshold={check['threshold']}"
        print(f"[{check['status']:4}] {check['name']}: {check['reason']}{detail}")
    print(f"RELEASE DECISION: {decision['decision']} ({decision['summary']['failed']} failed checks) -> {args.out}")
    return 0 if decision["decision"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
