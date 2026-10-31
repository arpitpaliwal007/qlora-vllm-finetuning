"""Promote a candidate artifact to the approved deployment slot, or roll back.

Promotion requires a PASS release decision that names the candidate's artifact
id; the candidate directory is re-hashed before copying so the promoted bytes
are provably the evaluated ones. The previously approved artifact is kept under
``previous-<timestamp>/`` for rollback.

    python -m qlora_vllm.promote --decision reports/release_decision.json \
        --candidate-dir artifacts/quantized-w4a16 --approved-dir deploy/approved
    python -m qlora_vllm.promote --rollback --approved-dir deploy/approved
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import read_json, utc_now, verify_artifact_dir, write_json
from .release_gate import parse_time

CURRENT = "current"
RELEASE_SUBDIR = "release"


def history_path(approved_dir: Path) -> Path:
    return approved_dir / "history.json"


def append_history(approved_dir: Path, entry: dict) -> None:
    path = history_path(approved_dir)
    history = read_json(path) if path.exists() else []
    history.append(entry)
    write_json(path, history)


def promote(decision_path: Path, candidate_dir: Path, approved_dir: Path, keep: int, max_age_hours: float) -> Path:
    decision = read_json(decision_path)
    if decision.get("decision") != "PASS":
        raise SystemExit(f"refusing to promote: release decision is {decision.get('decision')!r} ({decision.get('summary', {}).get('failed_checks')})")
    created = parse_time(decision.get("created_at"))
    if created is None or (datetime.now(UTC) - created).total_seconds() > max_age_hours * 3600:
        raise SystemExit("refusing to promote: release decision is missing a timestamp or is older than the allowed age")
    manifest = verify_artifact_dir(candidate_dir)  # raises if the files changed since the manifest was written
    if decision.get("candidate", {}).get("artifact_id") != manifest["artifact_id"]:
        raise SystemExit(
            f"refusing to promote: decision is for artifact {str(decision.get('candidate', {}).get('artifact_id'))[:12]}, "
            f"candidate dir is {manifest['artifact_id'][:12]}"
        )
    approved_dir.mkdir(parents=True, exist_ok=True)
    current = approved_dir / CURRENT
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    previous = None
    if current.exists():
        previous = approved_dir / f"previous-{stamp}"
        current.rename(previous)
    staging = approved_dir / f"incoming-{stamp}"
    shutil.copytree(candidate_dir, staging, ignore=shutil.ignore_patterns("checkpoint-*", "serve_status.json"))
    release_dir = staging / RELEASE_SUBDIR
    release_dir.mkdir()
    shutil.copy2(decision_path, release_dir / "release_decision.json")
    for label, report in (("results.json", decision["candidate"].get("report")), ("bench.json", decision.get("references", {}).get("bench", {}).get("report"))):
        if report and Path(report).exists():
            shutil.copy2(report, release_dir / label)
            predictions = Path(report).parent / "predictions.jsonl"
            if label == "results.json" and predictions.exists():
                shutil.copy2(predictions, release_dir / "predictions.jsonl")
    staging.rename(current)
    verify_artifact_dir(current)
    append_history(
        approved_dir,
        {
            "action": "promote",
            "at": utc_now(),
            "artifact_id": manifest["artifact_id"],
            "from": str(candidate_dir),
            "previous": str(previous) if previous else None,
            "decision": str(decision_path),
        },
    )
    _prune(approved_dir, keep)
    print(f"promoted {manifest['artifact_id'][:12]} -> {current}" + (f" (previous kept at {previous})" if previous else ""))
    return current


def rollback(approved_dir: Path) -> Path:
    previous = sorted(approved_dir.glob("previous-*"))
    if not previous:
        raise SystemExit("no previous approved artifact to roll back to")
    target = previous[-1]
    current = approved_dir / CURRENT
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    retired = None
    if current.exists():
        retired = approved_dir / f"rolledback-{stamp}"
        current.rename(retired)
    target.rename(current)
    manifest = verify_artifact_dir(current)
    append_history(
        approved_dir,
        {
            "action": "rollback",
            "at": utc_now(),
            "artifact_id": manifest["artifact_id"],
            "restored_from": str(target),
            "retired": str(retired) if retired else None,
        },
    )
    print(f"rolled back to {manifest['artifact_id'][:12]} (from {target.name})")
    return current


def _prune(approved_dir: Path, keep: int) -> None:
    for old in sorted(approved_dir.glob("previous-*"))[:-keep] if keep >= 0 else []:
        shutil.rmtree(old)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approved-dir", required=True)
    parser.add_argument("--decision")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--keep", type=int, default=2, help="how many previous artifacts to retain for rollback")
    parser.add_argument("--max-decision-age-hours", type=float, default=72)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)
    if args.rollback:
        rollback(Path(args.approved_dir))
        return 0
    if not (args.decision and args.candidate_dir):
        parser.error("--decision and --candidate-dir are required for promotion")
    promote(Path(args.decision), Path(args.candidate_dir), Path(args.approved_dir), args.keep, args.max_decision_age_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
