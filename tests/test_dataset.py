"""Chat templating, assistant-only loss masking and truncation policy (tiny CPU tokenizer)."""

from __future__ import annotations

import pytest

from qlora_vllm.dataset import IGNORE_INDEX, content_to_text, encode_dataset, encode_example, pad_collate
from qlora_vllm.prompting import build_messages

MESSAGES = build_messages("The Kell Bridge was opened in 1901 and spans 310 metres.", "When was the Kell Bridge opened?", "1901")


def test_content_to_text_handles_openai_text_parts():
    assert content_to_text([{"type": "text", "text": "hello"}, {"type": "image_url"}]) == "hello"


def test_loss_is_masked_on_prompt_and_supervised_on_assistant_turn(tiny_tokenizer):
    item = encode_example(MESSAGES, tiny_tokenizer, max_length=512)
    assert item["status"] == "ok"
    n_prompt = item["n_prompt_tokens"]
    assert item["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert item["labels"][n_prompt:] == item["input_ids"][n_prompt:]
    assert item["n_completion_tokens"] > 0
    prompt_text = tiny_tokenizer.decode(item["input_ids"][:n_prompt])
    completion_text = tiny_tokenizer.decode(item["input_ids"][n_prompt:])
    assert prompt_text.endswith("<|im_start|>assistant\n")
    assert completion_text.startswith("1901") and "<|im_end|>" in completion_text
    # the full sequence is the tokenizer's own rendering of the conversation
    assert tiny_tokenizer.decode(item["input_ids"]) == tiny_tokenizer.apply_chat_template(MESSAGES, tokenize=False)


def test_overlong_examples_are_dropped_or_raise_not_truncated(tiny_tokenizer):
    records = [{"id": "a", "messages": MESSAGES}, {"id": "b", "messages": build_messages("word " * 400, "q?", "word")}]
    encoded, stats = encode_dataset(records, tiny_tokenizer, max_length=200, on_too_long="drop")
    assert len(encoded) == 1 and stats.dropped_too_long == 1 and stats.kept == 1
    assert all(IGNORE_INDEX != label for label in encoded[0]["labels"][-3:])  # answer tokens still supervised
    with pytest.raises(ValueError, match="max_seq_length"):
        encode_dataset(records, tiny_tokenizer, max_length=200, on_too_long="error")
    with pytest.raises(ValueError):
        encode_dataset(records, tiny_tokenizer, max_length=200, on_too_long="truncate")


def test_example_must_end_with_assistant_turn(tiny_tokenizer):
    with pytest.raises(ValueError, match="assistant"):
        encode_example(MESSAGES[:-1], tiny_tokenizer, max_length=512)


def test_collate_pads_inputs_and_masks_padded_labels(tiny_tokenizer):
    import torch

    items, _ = encode_dataset([{"messages": MESSAGES}, {"messages": build_messages("Short.", "q?", "Short")}], tiny_tokenizer, 512)
    batch = pad_collate(tiny_tokenizer.pad_token_id)(items)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    short = min(len(i["input_ids"]) for i in items)
    row = 0 if len(items[0]["input_ids"]) == short else 1
    assert torch.all(batch["attention_mask"][row, short:] == 0)
    assert torch.all(batch["labels"][row, short:] == IGNORE_INDEX)
    assert torch.all(batch["input_ids"][row, short:] == tiny_tokenizer.pad_token_id)
