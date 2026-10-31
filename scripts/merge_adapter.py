#!/usr/bin/env python
"""Merge a trained LoRA adapter into its base model and write a merged checkpoint.

The adapter was trained against an NF4-quantized base; merging it into the
FP16/BF16 base is the standard export route but is not bit-identical to the
training-time model. The merged checkpoint is therefore evaluated in its own
right before it is quantized for serving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora_vllm.artifacts import read_artifact_manifest, write_artifact_manifest  # noqa: E402
from qlora_vllm.hardware import detect_hardware, resolve_precision, torch_dtype  # noqa: E402


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", help="defaults to the base recorded in the adapter manifest")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", default="auto", help="auto | fp16 | bf16 | fp32 (auto -> fp16 on T4)")
    args = parser.parse_args()

    adapter_manifest = read_artifact_manifest(args.adapter_dir)
    base_model = args.base_model or adapter_manifest["base_model"]
    revision = adapter_manifest.get("base_revision")
    hardware = detect_hardware()
    precision = resolve_precision(args.precision, hardware)
    dtype = torch_dtype(precision)
    print(f"[merge] base={base_model} revision={revision} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    manifest = write_artifact_manifest(
        args.output_dir,
        kind="merged_model",
        base_model=base_model,
        base_revision=revision,
        parent_artifact_id=adapter_manifest["artifact_id"],
        parent_kind="lora_adapter",
        dataset_hash=adapter_manifest.get("dataset_hash"),
        precision=str(dtype).replace("torch.", ""),
    )
    print(f"[merge] wrote {args.output_dir} artifact_id={manifest['artifact_id'][:12]}")


if __name__ == "__main__":
    main()
