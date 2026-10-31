"""Shared CPU fixtures: a tiny ChatML tokenizer and a tiny randomly initialised Qwen2 model.

Everything here is SYNTHETIC test scaffolding. The tiny model has no language
ability; it exists so the training/merge/evaluation code paths can execute on a
CPU without downloading anything. Nothing produced with it is model evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# ChatML template with the same system/user/assistant layout as Qwen2.5 (tool sections omitted).
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)


def build_tiny_tokenizer(save_dir: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    texts = []
    data = json.loads((FIXTURES / "squad_mini.json").read_text(encoding="utf-8"))["data"]
    for article in data:
        for para in article["paragraphs"]:
            texts.append(para["context"])
            texts.extend(qa["question"] for qa in para["qas"])
    texts.append("Answer the question using only the provided context. Context: Question:")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    )
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
    )
    fast.chat_template = CHATML_TEMPLATE
    fast.save_pretrained(save_dir)
    return fast


def build_tiny_model(save_dir: Path, tokenizer) -> None:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen2")
    tokenizer = build_tiny_tokenizer(path)
    build_tiny_model(path, tokenizer)
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture(scope="session")
def prepared_dataset(tmp_path_factory) -> Path:
    """Run the real prepare_data pipeline on the synthetic fixture (three-way split from one pool)."""
    from qlora_vllm.prepare_data import main

    out = tmp_path_factory.mktemp("prepared")
    main(
        [
            "--source",
            "local",
            "--train-json",
            str(FIXTURES / "squad_mini.json"),
            "--out",
            str(out),
            "--profile",
            "smoke",
            "--train-size",
            "18",
            "--validation-size",
            "6",
            "--test-size",
            "6",
            "--calibration-size",
            "4",
        ]
    )
    return out
