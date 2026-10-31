"""Data validation, normalization, dedup and leakage prevention (synthetic fixture, CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_vllm.prepare_data import (
    PrepConfig,
    clean_pool,
    contamination_report,
    grouped_take,
    iter_squad_json,
    normalize_text,
    read_jsonl,
    validate_record,
)
from qlora_vllm.prompting import SYSTEM_PROMPT

FIXTURE = Path(__file__).parent / "fixtures" / "squad_mini.json"
CFG = PrepConfig(train_size=18, validation_size=6, test_size=6, calibration_size=4)


def test_fixture_is_labelled_synthetic():
    assert "TEST FIXTURE" in json.loads(FIXTURE.read_text(encoding="utf-8"))["_note"]


def test_validate_rejects_malformed_records():
    ctx = "Alpha beta gamma delta epsilon zeta eta theta."
    ok, reason = validate_record({"id": "1", "title": "t", "context": ctx, "question": "q?", "answers": [{"text": "gamma", "answer_start": 11}]}, CFG)
    assert reason is None and ok["answers"] == ["gamma"]
    cases = {
        "empty_or_missing_question": {"question": " "},
        "malformed_answers": {"answers": []},
        "empty_answer_text": {"answers": [{"text": "", "answer_start": 0}]},
        "answer_span_mismatch": {"answers": [{"text": "gamma", "answer_start": 0}]},
        "answer_not_in_context_after_normalization": {"answers": [{"text": "omega"}]},
        "context_too_short": {"context": "tiny"},
    }
    for expected, override in cases.items():
        raw = {"id": "1", "title": "t", "context": ctx, "question": "q?", "answers": [{"text": "gamma", "answer_start": 11}], **override}
        assert validate_record(raw, CFG) == (None, expected), expected


def test_normalization_preserves_answers():
    ctx = "The Café René opened in 1957 and serves crème brûlée."
    assert normalize_text(ctx) == "The Café René opened in 1957 and serves crème brûlée."
    record, reason = validate_record(
        {"id": "u", "title": "Unicode", "context": ctx, "question": "What?", "answers": [{"text": "crème brûlée", "answer_start": ctx.index("crème")}]}, CFG
    )
    assert reason is None
    assert record["answers"] == ["crème brûlée"] and record["answers"][0] in record["context"]


def test_dedup_and_rejection_accounting():
    kept, audit = clean_pool(iter_squad_json(FIXTURE), CFG, "pool")
    reasons = audit["stats"]["rejection_reasons"]
    assert reasons["exact_duplicate"] == 1
    assert reasons["near_duplicate_question"] == 1
    assert reasons["answer_span_mismatch"] == 1
    assert reasons["empty_or_missing_question"] == 1
    assert reasons["malformed_answers"] == 1
    assert reasons["empty_answer_text"] == 1
    assert audit["stats"]["kept_records"] == 34  # 31 clean + 1 near-dup context record + 2 unicode
    assert {r["id"] for r in audit["rejected"]} == {"dup-exact", "dup-near", "bad-span", "bad-empty-q", "bad-no-answers", "bad-empty-answer"}
    # The near-duplicate context under a different title must share a group with the original.
    by_id = {r["id"]: r for r in kept}
    assert audit["stats"]["near_duplicate_context_pairs"] >= 1
    assert by_id["near-ctx-1"]["group"] == by_id["test-001"]["group"]
    # Every record keeps all of its reference answers.
    assert all(r["answers"] for r in kept)


def test_grouped_split_never_separates_a_group_and_is_deterministic():
    kept, _ = clean_pool(iter_squad_json(FIXTURE), CFG, "pool")
    sizes = {"test": 6, "validation": 6, "train": 18}
    a = grouped_take(kept, sizes, seed=1)
    b = grouped_take(kept, sizes, seed=1)
    assert {k: [r["id"] for r in v] for k, v in a.items()} == {k: [r["id"] for r in v] for k, v in b.items()}
    groups = {name: {r["group"] for r in rows} for name, rows in a.items()}
    assert not (groups["test"] & groups["train"]) and not (groups["test"] & groups["validation"]) and not (groups["validation"] & groups["train"])
    assert contamination_report(a)["passed"]
    assert grouped_take(kept, sizes, seed=2) != a  # the seed matters


def test_contamination_report_catches_leaks():
    rec = {"id": "x", "title": "T", "context": "Some shared context here.", "question": "q?", "group": "g", "dedup_key": "k"}
    report = contamination_report({"train": [rec], "test": [dict(rec, id="y", question="other?", dedup_key="k2")]})
    assert not report["passed"]
    assert report["pairs"]["train/test"]["context_overlap"] == 1 and report["pairs"]["train/test"]["title_overlap"] == 1


def test_prepared_outputs_and_manifest(prepared_dataset: Path):
    manifest = json.loads((prepared_dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contamination_check"]["passed"]
    # whole groups are assigned, then trimmed; train receives whatever groups remain (<= its target)
    assert 0 < manifest["splits"]["train"]["records"] <= 18
    assert manifest["splits"]["validation"]["records"] == 6
    assert manifest["splits"]["test"]["records"] == 6
    assert manifest["splits"]["calibration"]["subset_of"] == "train"
    assert len(manifest["dataset_hash"]) == 64
    assert manifest["pools"]["train_pool"]["rejection_reasons"]["exact_duplicate"] == 1
    rejections = json.loads((prepared_dataset / "rejections.json").read_text(encoding="utf-8"))
    assert any(r["reason"] == "near_duplicate_question" for r in rejections["train_pool"])
    train, test, calib = (read_jsonl(prepared_dataset / f"{n}.jsonl") for n in ("train", "test", "calibration"))
    assert {r["id"] for r in calib} <= {r["id"] for r in train}
    assert not ({r["context"] for r in train} & {r["context"] for r in test})
    assert not ({r["title"] for r in train} & {r["title"] for r in test})
    sample = train[0]
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    assert sample["messages"][0]["content"] == SYSTEM_PROMPT
    assert sample["messages"][2]["content"] == sample["answers"][0]
    assert sample["context"] in sample["messages"][1]["content"] and sample["question"] in sample["messages"][1]["content"]


def test_official_validation_pool_becomes_test(tmp_path: Path):
    from qlora_vllm.prepare_data import prepare

    pool = list(iter_squad_json(FIXTURE))
    train_pool = [r for r in pool if r["title"] not in {"Test Article Festival", "Test Article Unicode"}]
    # The test pool contains one overlapping article (Rivers) that must be excluded from the test split.
    test_pool = [r for r in pool if r["title"] in {"Test Article Festival", "Test Article Unicode", "Test Article Rivers"}]
    manifest = prepare(train_pool, test_pool, PrepConfig(train_size=12, validation_size=4, test_size=5, calibration_size=2), tmp_path, {"name": "fixture"})
    assert manifest["contamination_check"]["passed"]
    assert manifest["pools"]["test_pool"]["excluded_overlapping_train_pool"] > 0
    test = read_jsonl(tmp_path / "test.jsonl")
    assert test and all(r["title"] in {"Test Article Festival", "Test Article Unicode"} for r in test)


@pytest.mark.parametrize("bad", [{"train_size": 18, "validation_size": 0}])
def test_zero_sized_split_is_allowed(bad):
    kept, _ = clean_pool(iter_squad_json(FIXTURE), CFG, "pool")
    out = grouped_take(kept, {"validation": 0, "train": 18}, seed=0)
    assert out["validation"] == [] and len(out["train"]) == 18
