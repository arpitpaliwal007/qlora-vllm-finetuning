#!/usr/bin/env python
"""Quantize the merged fine-tuned model to W4A16 (GPTQ, group size 128) with llm-compressor.

Why this route: vLLM loads ``compressed-tensors`` checkpoints natively and lists
GPTQ/AWQ INT4 as supported on Turing (T4, compute capability 7.5), whereas FP8
is not. llm-compressor is maintained by the vLLM project, so the checkpoint
format and the serving engine stay in step. The GGUF helper remains for
llama.cpp/Ollama-style deployments but is not the gated serving path.

Calibration data comes from ``calibration.jsonl`` written by prepare_data
(a seeded sample of the *train* split, never validation/test); its sha256 is
recorded in the artifact manifest.

    python scripts/quantize_w4a16.py --model-dir artifacts/merged-model \
        --calibration data/squad_qa/calibration.jsonl --output-dir artifacts/quantized-w4a16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora_vllm.artifacts import read_artifact_manifest, sha256_file, write_artifact_manifest  # noqa: E402
from qlora_vllm.hardware import detect_hardware  # noqa: E402
from qlora_vllm.prepare_data import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="merged fine-tuned model directory (with artifact_manifest.json)")
    parser.add_argument("--calibration", required=True, help="calibration.jsonl from prepare_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--scheme", default="W4A16")
    args = parser.parse_args()

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parent = read_artifact_manifest(args.model_dir)
    hardware = detect_hardware()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="auto", device_map={"": 0} if hardware["cuda_available"] else None)

    records = read_jsonl(Path(args.calibration))[: args.num_samples]
    # Calibrate on full conversations rendered with the model's own chat template, i.e. the
    # same token distribution the adapter was trained on.
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in records]
    encoded = [tokenizer(t, max_length=args.max_seq_length, truncation=True, add_special_tokens=False) for t in texts]
    dataset = Dataset.from_list([{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in encoded])

    recipe = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"])
    t0 = time.time()
    oneshot(model=model, dataset=dataset, recipe=recipe, max_seq_length=args.max_seq_length, num_calibration_samples=len(dataset))
    seconds = time.time() - t0
    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    manifest = write_artifact_manifest(
        args.output_dir,
        kind="quantized_w4a16_compressed_tensors",
        base_model=parent.get("base_model"),
        base_revision=parent.get("base_revision"),
        parent_artifact_id=parent["artifact_id"],
        parent_kind=parent["kind"],
        dataset_hash=parent.get("dataset_hash"),
        quantization={
            "tool": "llm-compressor",
            "algorithm": "GPTQ",
            "scheme": args.scheme,
            "targets": "Linear",
            "ignore": ["lm_head"],
            "calibration_file": str(args.calibration),
            "calibration_sha256": sha256_file(Path(args.calibration)),
            "calibration_samples": len(dataset),
            "calibration_max_seq_length": args.max_seq_length,
            "calibration_source": "seeded sample of the prepared train split (see dataset manifest)",
            "seconds": round(seconds, 1),
        },
        hardware=hardware,
    )
    print(f"[quantize] wrote {args.output_dir} artifact_id={manifest['artifact_id'][:12]} in {seconds:.0f}s")


if __name__ == "__main__":
    main()
