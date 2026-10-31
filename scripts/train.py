#!/usr/bin/env python
"""Fine-tune a causal LM with QLoRA (NF4 base + LoRA adapters).

The trainer is ``transformers.Trainer`` with PEFT rather than TRL's SFTTrainer:
this project needs an exact assistant-only loss mask (Qwen2.5's chat template
has no ``{% generation %}`` markers, which TRL's ``assistant_only_loss``
requires) and an explicit policy for over-long examples instead of TRL's
default right-truncation, which would silently delete the supervised answer.

Outputs under ``output_dir``: the LoRA adapter, tokenizer, Trainer checkpoints,
``training_metadata.json`` and ``artifact_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora_vllm.artifacts import read_json, sha256_file, utc_now, write_artifact_manifest, write_json  # noqa: E402
from qlora_vllm.config import load_config  # noqa: E402
from qlora_vllm.dataset import encode_dataset, pad_collate  # noqa: E402
from qlora_vllm.hardware import detect_hardware, resolve_precision, torch_dtype  # noqa: E402
from qlora_vllm.prepare_data import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with QLoRA.")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument("--max-steps", type=int, help="override max_steps (useful for smoke runs)")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    return parser.parse_args()


def dataset_provenance(train_path: Path) -> dict:
    """Hash the train file and pick up the dataset manifest if the file came from prepare_data."""
    info = {"train_path": str(train_path), "train_sha256": sha256_file(train_path)}
    manifest_path = train_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        info["dataset_hash"] = manifest.get("dataset_hash")
        info["manifest_path"] = str(manifest_path)
    return info


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    set_seed(cfg.seed)
    hardware = detect_hardware()
    precision = resolve_precision(cfg.precision, hardware)
    compute_dtype = torch_dtype(precision)
    use_cuda = bool(hardware["cuda_available"])
    if cfg.load_in_4bit and not use_cuda:
        raise RuntimeError("load_in_4bit requires a CUDA GPU; set load_in_4bit: false for CPU runs")
    print(f"[train] precision={precision} cuda={use_cuda} gpus={[g['name'] for g in hardware['gpus']]}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {"dtype": compute_dtype, "revision": cfg.model_revision}
    if cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if use_cuda:
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    model.config.use_cache = False
    gc_kwargs = {"use_reentrant": False}
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing, gradient_checkpointing_kwargs=gc_kwargs)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(Path(cfg.dataset_path))
    if cfg.validation_path:
        eval_records = read_jsonl(Path(cfg.validation_path))
    elif cfg.validation_split:
        import random

        random.Random(cfg.seed).shuffle(train_records)
        n_eval = max(1, int(len(train_records) * cfg.validation_split))
        eval_records, train_records = train_records[:n_eval], train_records[n_eval:]
    else:
        eval_records = []
    train_dataset, train_stats = encode_dataset(train_records, tokenizer, cfg.max_seq_length, cfg.on_too_long)
    eval_dataset, eval_stats = encode_dataset(eval_records, tokenizer, cfg.max_seq_length, cfg.on_too_long)
    print(f"[train] train examples: {train_stats.as_dict()}")
    print(f"[train] eval examples: {eval_stats.as_dict()}")
    if not train_dataset:
        raise RuntimeError("no training examples left after encoding")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_ratio,  # float in [0, 1) is interpreted as a ratio of total steps
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.eval_steps if eval_dataset else None,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        optim="paged_adamw_32bit" if cfg.load_in_4bit else "adamw_torch",
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset or None,
        data_collator=pad_collate(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    resume = cfg.resume_from_checkpoint
    if args.resume or resume is True or resume == "auto":
        resume = get_last_checkpoint(str(output_dir))
        print(f"[train] resuming from {resume}")
    started = utc_now()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume or None)
    train_seconds = time.time() - t0
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(result.metrics)
    if eval_dataset:
        metrics.update(trainer.evaluate())
    provenance = dataset_provenance(Path(cfg.dataset_path))
    metadata = {
        "schema_version": 1,
        "started_at": started,
        "finished_at": utc_now(),
        "train_seconds": round(train_seconds, 1),
        "model": {
            "name": cfg.model_name,
            "requested_revision": cfg.model_revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
        },
        "dataset": {**provenance, "validation_path": cfg.validation_path},
        "encoding": {"train": train_stats.as_dict(), "validation": eval_stats.as_dict(), "max_seq_length": cfg.max_seq_length},
        "config": cfg.as_dict(),
        "precision": precision,
        "resumed_from": resume or None,
        "hardware": hardware,
        "metrics": metrics,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    write_json(output_dir / "training_metadata.json", metadata)
    write_artifact_manifest(
        output_dir,
        kind="lora_adapter",
        base_model=cfg.model_name,
        base_revision=metadata["model"]["resolved_revision"] or cfg.model_revision,
        dataset_hash=provenance.get("dataset_hash"),
        train_sha256=provenance["train_sha256"],
        training_metadata="training_metadata.json",
    )
    print(json.dumps({"train_seconds": round(train_seconds, 1), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
