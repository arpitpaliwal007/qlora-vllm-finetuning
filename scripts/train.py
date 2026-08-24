#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from qlora_vllm.config import load_config
from qlora_vllm.dataset import format_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with QLoRA.")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config if cfg.load_in_4bit else None,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    raw = load_dataset("json", data_files=cfg.dataset_path, split="train")
    if "messages" not in raw.column_names:
        raise ValueError("Dataset must contain a 'messages' field on every JSONL row.")
    dataset = raw.map(lambda row: format_example(row, tokenizer), remove_columns=raw.column_names)
    split = dataset.train_test_split(test_size=cfg.validation_split, seed=cfg.seed) if cfg.validation_split else None
    train_dataset = split["train"] if split else dataset
    eval_dataset = split["test"] if split else None

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_steps=cfg.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="paged_adamw_32bit" if cfg.load_in_4bit else "adamw_torch",
        report_to="none",
        seed=cfg.seed,
    )
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
