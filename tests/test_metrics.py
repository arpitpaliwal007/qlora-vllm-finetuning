from __future__ import annotations

import pytest

from qlora_vllm.metrics import exact_match, normalize_answer, paired_bootstrap, score_prediction, token_f1


def test_normalization_matches_squad_rules():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("an answer, a question.") == "answer question"


def test_exact_match_and_f1():
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0
    assert exact_match("Eiffel", "eiffel tower") == 0.0
    assert token_f1("Eiffel", "eiffel tower") == pytest.approx(2 / 3)
    assert token_f1("", "x") == 0.0 and token_f1("", "") == 1.0
    assert token_f1("completely wrong", "eiffel tower") == 0.0


def test_best_over_multiple_references():
    scores = score_prediction("The 1889", ["1889", "the year 1889", "1889 for the World's Fair"])
    assert scores["exact_match"] == 1.0 and scores["f1"] == 1.0
    scores = score_prediction("330 metres", ["330", "330 metres tall"])
    assert scores["exact_match"] == 0.0 and scores["f1"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        score_prediction("x", [])


def test_paired_bootstrap_detects_improvement_and_null():
    a = [0.0] * 50 + [1.0] * 50
    b = [0.0] * 30 + [1.0] * 70
    res = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert res["mean_diff"] == pytest.approx(0.2)
    assert res["ci_low"] > 0.0 and res["ci_high"] > res["ci_low"]
    same = paired_bootstrap(a, a, n_resamples=200, seed=1)
    assert same["mean_diff"] == 0.0 and same["ci_low"] == 0.0 == same["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
