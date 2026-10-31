"""SQuAD-style answer metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles and collapse whitespace (official SQuAD normalization)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_references(metric, prediction: str, references: Sequence[str]) -> float:
    if not references:
        raise ValueError("at least one reference answer is required")
    return max(metric(prediction, ref) for ref in references)


def score_prediction(prediction: str, references: Sequence[str]) -> dict[str, float]:
    return {
        "exact_match": best_over_references(exact_match, prediction, references),
        "f1": best_over_references(token_f1, prediction, references),
    }


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for ``mean(b) - mean(a)`` over the same examples (paired)."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap needs two equally sized, non-empty score lists")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "n": int(diff.size),
        "n_resamples": n_resamples,
        "p_diff_le_0": float((means <= 0).mean()),
    }
