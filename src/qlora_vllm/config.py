from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    model_revision: str | None = None
    validation_path: str | None = None
    validation_split: float = 0.0
    max_seq_length: int = 1024
    on_too_long: str = "drop"  # drop | error (never truncate away the answer)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ()
    num_train_epochs: float = 2
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    eval_steps: int = 100
    seed: int = 42
    precision: str = "auto"  # auto | fp16 | bf16 | fp32
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    raw["lora_target_modules"] = tuple(raw.get("lora_target_modules", ()))
    # Legacy keys from the original project: bf16/fp16 booleans map onto `precision`.
    legacy_bf16, legacy_fp16 = raw.pop("bf16", None), raw.pop("fp16", None)
    if "precision" not in raw:
        raw["precision"] = "bf16" if legacy_bf16 else "fp16" if legacy_fp16 else "auto"
    unknown = set(raw) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    cfg = TrainConfig(**raw)
    if cfg.on_too_long not in {"drop", "error"}:
        raise ValueError("on_too_long must be 'drop' or 'error'")
    return cfg
