"""Chat-format dataset helpers: templating, assistant-only loss masking and truncation policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

IGNORE_INDEX = -100


def content_to_text(content: Any) -> str:
    """Normalize OpenAI-style text parts to the string expected by chat templates."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError("message content must be a string or a list of text parts")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": content_to_text(m["content"])} for m in messages]


def format_example(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Render a full conversation as text (kept for compatibility with the original project)."""
    return {"text": tokenizer.apply_chat_template(normalize_messages(example["messages"]), tokenize=False)}


def encode_example(messages: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Tokenize one conversation with loss only on the final assistant turn.

    The prompt (every turn before the last assistant message, rendered with
    ``add_generation_prompt=True``) is tokenized separately from the assistant
    completion so the mask boundary is exact rather than inferred by searching
    for a token sequence. Returns ``status`` of ``ok`` or ``too_long``; a
    ``too_long`` example is never silently truncated, because right-truncation
    would delete the supervised answer and left-truncation the context.
    """
    messages = normalize_messages(messages)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("each training example must end with an assistant message")
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template does not render the prompt as a prefix of the full conversation")
    completion_text = full_text[len(prompt_text) :]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError("assistant completion tokenized to zero tokens")
    input_ids = prompt_ids + completion_ids
    status = "ok" if len(input_ids) <= max_length else "too_long"
    return {
        "input_ids": input_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
        "attention_mask": [1] * len(input_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(completion_ids),
        "status": status,
    }


@dataclass
class EncodingStats:
    total: int = 0
    kept: int = 0
    dropped_too_long: int = 0
    max_tokens: int = 0
    total_tokens: int = 0
    total_completion_tokens: int = 0
    status_counts: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_too_long": self.dropped_too_long,
            "max_tokens": self.max_tokens,
            "mean_tokens": round(self.total_tokens / self.kept, 1) if self.kept else 0,
            "mean_completion_tokens": round(self.total_completion_tokens / self.kept, 1) if self.kept else 0,
        }


def encode_dataset(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    on_too_long: str = "drop",
) -> tuple[list[dict[str, list[int]]], EncodingStats]:
    """Encode records; ``on_too_long`` is ``drop`` (count and skip) or ``error`` (abort)."""
    if on_too_long not in {"drop", "error"}:
        raise ValueError("on_too_long must be 'drop' or 'error'")
    stats = EncodingStats()
    encoded: list[dict[str, list[int]]] = []
    for record in records:
        stats.total += 1
        item = encode_example(record["messages"], tokenizer, max_length)
        if item["status"] == "too_long":
            if on_too_long == "error":
                raise ValueError(f"example {record.get('id', stats.total)} has {len(item['input_ids'])} tokens > max_seq_length={max_length}")
            stats.dropped_too_long += 1
            continue
        stats.kept += 1
        stats.max_tokens = max(stats.max_tokens, len(item["input_ids"]))
        stats.total_tokens += len(item["input_ids"])
        stats.total_completion_tokens += item["n_completion_tokens"]
        encoded.append({k: item[k] for k in ("input_ids", "labels", "attention_mask")})
    return encoded, stats


def pad_collate(pad_token_id: int):
    """Return a collate function that right-pads ``input_ids`` and masks padded labels."""
    import torch

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)

        def pad(key: str, value: int) -> Any:
            return torch.tensor([item[key] + [value] * (width - len(item[key])) for item in batch], dtype=torch.long)

        return {
            "input_ids": pad("input_ids", pad_token_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE_INDEX),
        }

    return collate
