from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    validation_split: float = 0.05
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ()
    num_train_epochs: float = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    eval_steps: int = 100
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    raw["lora_target_modules"] = tuple(raw.get("lora_target_modules", ()))
    return TrainConfig(**raw)
