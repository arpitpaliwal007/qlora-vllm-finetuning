"""Prepare a grounded question-answering dataset (SQuAD v1.1 schema) for QLoRA training.

Pipeline: validate -> normalize -> exact/near dedup -> leakage-safe grouped split ->
chat-format JSONL + manifest. Groups are Wikipedia article titles merged with
near-duplicate context clusters, so no context ever appears in two splits.

Usage::

    python -m qlora_vllm.prepare_data --source hf --out data/squad_qa --profile full
    python -m qlora_vllm.prepare_data --source local --train-json train-v1.1.json \
        --test-json dev-v1.1.json --out data/squad_qa --profile smoke

The near-duplicate index is adapted from the author's
``llm-data-curation-tokenization`` repository (MinHash + LSH banding).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from hashlib import blake2b, sha256
from pathlib import Path
from typing import Any

from .artifacts import package_versions, sha256_file, utc_now, write_json
from .prompting import build_messages

SQUAD = {
    "name": "rajpurkar/squad",
    "config": "plain_text",
    "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "license": "cc-by-sa-4.0",
    "url": "https://huggingface.co/datasets/rajpurkar/squad",
    "paper": "Rajpurkar et al., 2016 (SQuAD v1.1)",
}
PROFILES = {
    # Tiny profile for CPU/pipeline smoke tests on real data.
    "smoke": {"train_size": 256, "validation_size": 64, "test_size": 64, "calibration_size": 32},
    # Practical single-T4 run: ~2 epochs over 20k examples is a few GPU hours.
    "full": {"train_size": 20000, "validation_size": 1000, "test_size": 1000, "calibration_size": 256},
}
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


@dataclass
class PrepConfig:
    train_size: int
    validation_size: int
    test_size: int
    calibration_size: int
    seed: int = 42
    max_context_chars: int = 6000
    min_context_chars: int = 20
    near_dup_question_jaccard: float = 0.9
    near_dup_context_jaccard: float = 0.8
    minhash_permutations: int = 64
    lsh_bands: int = 16
    shingle_size: int = 3
    min_groups_per_split: int = 8


# --------------------------------------------------------------------------- loading


def iter_squad_json(path: Path) -> Iterator[dict[str, Any]]:
    """Yield flat records from the official nested SQuAD v1.1 JSON, or from a flat JSONL file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield {
                    "id": qa.get("id"),
                    "title": article.get("title"),
                    "context": paragraph.get("context"),
                    "question": qa.get("question"),
                    "answers": qa.get("answers"),
                }


def load_hf_squad() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(SQUAD["name"], SQUAD["config"], revision=SQUAD["revision"])
    return list(ds["train"]), list(ds["validation"])


# --------------------------------------------------------------------------- validation


def normalize_text(text: str) -> str:
    """NFKC-normalize and collapse whitespace. Applied identically to contexts, questions and answers."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _answer_pairs(raw: Any) -> list[tuple[str, int | None]]:
    """Accept ``[{"text","answer_start"}]`` or ``{"text": [...], "answer_start": [...]}``."""
    if isinstance(raw, dict):
        texts = raw.get("text") or []
        starts = raw.get("answer_start") or [None] * len(texts)
        return list(zip(texts, starts, strict=False))
    if isinstance(raw, list):
        return [(a.get("text"), a.get("answer_start")) for a in raw if isinstance(a, dict)]
    return []


def validate_record(raw: dict[str, Any], cfg: PrepConfig) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(clean_record, None)`` or ``(None, rejection_reason)``."""
    for field in ("id", "title", "context", "question"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, f"empty_or_missing_{field}"
    context, question = raw["context"], raw["question"]
    if len(context) > cfg.max_context_chars:
        return None, "context_too_long"
    if len(context.strip()) < cfg.min_context_chars:
        return None, "context_too_short"
    pairs = _answer_pairs(raw.get("answers"))
    if not pairs:
        return None, "malformed_answers"
    answers: list[str] = []
    for text, start in pairs:
        if not isinstance(text, str) or not text.strip():
            return None, "empty_answer_text"
        if start is not None and context[start : start + len(text)] != text:
            return None, "answer_span_mismatch"
        norm = normalize_text(text)
        if norm not in normalize_text(context):
            return None, "answer_not_in_context_after_normalization"
        if norm not in answers:
            answers.append(norm)
    return (
        {
            "id": str(raw["id"]),
            "title": normalize_text(raw["title"]),
            "context": normalize_text(context),
            "question": normalize_text(question),
            "answers": answers,
        },
        None,
    )


# --------------------------------------------------------------------------- dedup


def dedup_key(context: str, question: str) -> str:
    return sha256(f"{context.lower()}\n{question.lower()}".encode()).hexdigest()


def tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def shingles(text: str, size: int) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _hash(value: str, seed: int) -> int:
    return int.from_bytes(blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big")


class NearDuplicateIndex:
    """MinHash/LSH index returning near-duplicate ids for a set of shingles."""

    def __init__(self, permutations: int, bands: int, threshold: float) -> None:
        if permutations % bands:
            raise ValueError("permutations must be divisible by bands")
        self.rows, self.bands, self.threshold = permutations // bands, bands, threshold
        self.permutations = permutations
        self.buckets: dict[tuple[int, tuple[int, ...]], set[str]] = {}
        self.items: dict[str, set[str]] = {}

    def _signature(self, items: set[str]) -> tuple[int, ...]:
        return tuple(min(_hash(item, seed) for item in items) for seed in range(self.permutations))

    def query(self, items: set[str]) -> list[str]:
        signature = self._signature(items)
        candidates: set[str] = set()
        for band in range(self.bands):
            candidates.update(self.buckets.get((band, signature[band * self.rows : (band + 1) * self.rows]), ()))
        return [c for c in candidates if jaccard(items, self.items[c]) >= self.threshold]

    def add(self, key: str, items: set[str]) -> None:
        signature = self._signature(items)
        self.items[key] = items
        for band in range(self.bands):
            self.buckets.setdefault((band, signature[band * self.rows : (band + 1) * self.rows]), set()).add(key)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def clean_pool(raw_records: Iterable[dict[str, Any]], cfg: PrepConfig, pool_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, normalize and deduplicate one pool of records.

    Exact duplicates (same normalized context + question) are rejected. Questions
    that are near-duplicates of an earlier question on the same context are
    rejected. Near-duplicate *contexts* are kept but merged into one group so the
    splitter never separates them.
    """
    reasons: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    kept: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    questions_by_context: dict[str, list[set[str]]] = defaultdict(list)
    context_index = NearDuplicateIndex(cfg.minhash_permutations, cfg.lsh_bands, cfg.near_dup_context_jaccard)
    context_hash_seen: set[str] = set()
    groups = UnionFind()
    near_context_pairs = 0
    total = 0
    for raw in raw_records:
        total += 1
        record, reason = validate_record(raw, cfg)
        if record is None:
            reasons[reason] += 1
            rejected.append({"id": str(raw.get("id")), "reason": reason})
            continue
        key = dedup_key(record["context"], record["question"])
        if key in seen_keys:
            reasons["exact_duplicate"] += 1
            rejected.append({"id": record["id"], "reason": "exact_duplicate"})
            continue
        context_hash = sha256(record["context"].lower().encode()).hexdigest()
        question_tokens = tokens(record["question"])
        if any(jaccard(question_tokens, prior) >= cfg.near_dup_question_jaccard for prior in questions_by_context[context_hash]):
            reasons["near_duplicate_question"] += 1
            rejected.append({"id": record["id"], "reason": "near_duplicate_question"})
            continue
        seen_keys.add(key)
        questions_by_context[context_hash].append(question_tokens)
        group = f"title:{record['title'].lower()}"
        if context_hash not in context_hash_seen:
            context_hash_seen.add(context_hash)
            items = shingles(record["context"], cfg.shingle_size)
            for other in context_index.query(items):
                near_context_pairs += 1
                groups.union(f"ctx:{context_hash}", f"ctx:{other}")
            context_index.add(context_hash, items)
        groups.union(group, f"ctx:{context_hash}")
        record["group"] = f"ctx:{context_hash}"
        record["dedup_key"] = key
        kept.append(record)
    for record in kept:
        record["group"] = groups.find(record["group"])
    stats = {
        "pool": pool_name,
        "input_records": total,
        "kept_records": len(kept),
        "rejected_records": len(rejected),
        "rejection_reasons": dict(reasons),
        "unique_contexts": len(context_hash_seen),
        "near_duplicate_context_pairs": near_context_pairs,
        "groups": len({r["group"] for r in kept}),
    }
    return kept, {"stats": stats, "rejected": rejected}


# --------------------------------------------------------------------------- splitting


def grouped_take(records: list[dict[str, Any]], sizes: dict[str, int], seed: int, min_groups: int = 8) -> dict[str, list[dict[str, Any]]]:
    """Assign whole groups to the named splits in order until each size is met.

    Groups are shuffled deterministically and a group is never split across two
    outputs. To keep a split from being filled by one large article, at most
    ``ceil(size / min_groups)`` records of each group are used; the rest of that
    group is discarded, never given to another split, so trimming cannot leak.
    """
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)
    group_ids = sorted(by_group)
    random.Random(seed).shuffle(group_ids)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    cursor = 0
    for name, target in sizes.items():
        per_group = max(1, -(-target // max(1, min_groups)))
        while len(out[name]) < target and cursor < len(group_ids):
            out[name].extend(by_group[group_ids[cursor]][:per_group])
            cursor += 1
        out[name] = out[name][:target]
    return out


def contamination_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = [n for n in ("train", "validation", "test") if n in splits]
    report: dict[str, Any] = {"pairs": {}, "passed": True}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            left, right = splits[a], splits[b]
            overlaps = {
                "title_overlap": len({r["title"].lower() for r in left} & {r["title"].lower() for r in right}),
                "context_overlap": len({r["context"].lower() for r in left} & {r["context"].lower() for r in right}),
                "group_overlap": len({r["group"] for r in left} & {r["group"] for r in right}),
                "question_context_overlap": len({r["dedup_key"] for r in left} & {r["dedup_key"] for r in right}),
            }
            report["pairs"][f"{a}/{b}"] = overlaps
            if any(overlaps.values()):
                report["passed"] = False
    return report


# --------------------------------------------------------------------------- output


def to_output_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "title": record["title"],
        "context": record["context"],
        "question": record["question"],
        "answers": record["answers"],
        "messages": build_messages(record["context"], record["question"], record["answers"][0]),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(
    train_pool: list[dict[str, Any]],
    test_pool: list[dict[str, Any]] | None,
    cfg: PrepConfig,
    out_dir: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    train_clean, train_audit = clean_pool(train_pool, cfg, "train_pool")
    audits = {"train_pool": train_audit}
    if test_pool is None:
        # Three-way grouped split from a single pool.
        splits = grouped_take(
            train_clean,
            {"test": cfg.test_size, "validation": cfg.validation_size, "train": cfg.train_size},
            cfg.seed,
            cfg.min_groups_per_split,
        )
        split_method = (
            "single pool; whole title/near-duplicate-context groups assigned to test, validation, train; at most "
            f"ceil(size/{cfg.min_groups_per_split}) records used per group, the remainder of a group is discarded"
        )
    else:
        test_clean, test_audit = clean_pool(test_pool, cfg, "test_pool")
        audits["test_pool"] = test_audit
        # A test record whose group/title/context also occurs in the train pool is
        # excluded from the test set so the held-out set is disjoint by construction.
        train_titles = {r["title"].lower() for r in train_clean}
        train_contexts = {r["context"].lower() for r in train_clean}
        eligible = [r for r in test_clean if r["title"].lower() not in train_titles and r["context"].lower() not in train_contexts]
        excluded_from_test = len(test_clean) - len(eligible)
        splits = grouped_take(eligible, {"test": cfg.test_size}, cfg.seed, cfg.min_groups_per_split)
        splits.update(grouped_take(train_clean, {"validation": cfg.validation_size, "train": cfg.train_size}, cfg.seed, cfg.min_groups_per_split))
        audits["test_pool"]["stats"]["excluded_overlapping_train_pool"] = excluded_from_test
        split_method = (
            "test drawn from the separate test pool (SQuAD official validation split) after removing any record "
            "sharing a title or context with the train pool; validation/train drawn from the train pool by whole "
            f"title/near-duplicate-context groups; at most ceil(size/{cfg.min_groups_per_split}) records used per group"
        )
    contamination = contamination_report(splits)
    if not contamination["passed"]:
        raise RuntimeError(f"contamination check failed: {contamination}")
    rng = random.Random(cfg.seed + 1)
    calibration = rng.sample(splits["train"], min(cfg.calibration_size, len(splits["train"])))

    files: dict[str, dict[str, Any]] = {}
    for name, records in [*splits.items(), ("calibration", calibration)]:
        path = out_dir / f"{name}.jsonl"
        count = write_jsonl(path, (to_output_record(r) for r in records))
        files[name] = {
            "path": path.name,
            "records": count,
            "groups": len({r["group"] for r in records}),
            "sha256": sha256_file(path),
        }
    files["calibration"]["subset_of"] = "train"
    files["calibration"]["note"] = "quantization calibration data; sampled from train only, never from validation/test"
    write_json(out_dir / "rejections.json", {k: v["rejected"] for k, v in audits.items()})
    dataset_hash = sha256("|".join(files[n]["sha256"] for n in ("train", "validation", "test")).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "task": "grounded extractive question answering (context + question -> answer span)",
        "source": source,
        "parameters": asdict(cfg),
        "normalization": "Unicode NFKC + whitespace collapse on title/context/question/answers; each normalized "
        "answer re-verified as a substring of the normalized context, otherwise the record is rejected",
        "deduplication": "exact duplicates on normalized (context, question); near-duplicate questions on the same "
        f"context rejected at token-Jaccard >= {cfg.near_dup_question_jaccard}; near-duplicate contexts "
        f"(MinHash/LSH, {cfg.shingle_size}-gram Jaccard >= {cfg.near_dup_context_jaccard}) merged into one split group",
        "split_method": split_method,
        "pools": {k: v["stats"] for k, v in audits.items()},
        "splits": files,
        "contamination_check": contamination,
        "dataset_hash": dataset_hash,
        "packages": package_versions(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["hf", "local"], required=True)
    parser.add_argument("--train-json", help="local SQuAD v1.1 JSON (or flat JSONL) used as the train pool")
    parser.add_argument("--test-json", help="optional local pool reserved for the test split (e.g. dev-v1.1.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-groups-per-split", type=int, default=8, help="spread each split over at least this many title groups")
    for key in ("train_size", "validation_size", "test_size", "calibration_size"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int)
    args = parser.parse_args(argv)

    sizes = dict(PROFILES[args.profile])
    for key in list(sizes):
        if getattr(args, key) is not None:
            sizes[key] = getattr(args, key)
    cfg = PrepConfig(seed=args.seed, min_groups_per_split=args.min_groups_per_split, **sizes)

    if args.source == "hf":
        train_pool, test_pool = load_hf_squad()
        source: dict[str, Any] = {**SQUAD, "test_pool": "official validation split"}
    else:
        if not args.train_json:
            parser.error("--train-json is required with --source local")
        train_pool = list(iter_squad_json(Path(args.train_json)))
        test_pool = list(iter_squad_json(Path(args.test_json))) if args.test_json else None
        source = {
            "name": "local SQuAD-format files",
            "files": {str(p): sha256_file(Path(p)) for p in [args.train_json, args.test_json] if p},
            "license": "see upstream dataset",
        }
    manifest = prepare(train_pool, test_pool, cfg, Path(args.out), source)
    print(json.dumps({k: manifest[k] for k in ("splits", "contamination_check", "dataset_hash")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
