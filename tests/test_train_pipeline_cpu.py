"""End-to-end CPU exercise of train -> resume -> merge -> evaluate -> compare on a tiny random model.

This proves the code paths run and produce correctly linked artifacts/reports.
It is NOT GPU validation: the model is random, NF4 loading is skipped (CPU) and
no number produced here is a model result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"command failed: {args}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    return proc


def test_train_resume_merge_evaluate(tiny_model_dir: Path, prepared_dataset: Path, tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    config = {
        "model_name": str(tiny_model_dir),
        "dataset_path": str(prepared_dataset / "train.jsonl"),
        "validation_path": str(prepared_dataset / "validation.jsonl"),
        "output_dir": str(adapter_dir),
        "max_seq_length": 256,
        "load_in_4bit": False,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_target_modules": ["q_proj", "v_proj"],
        "max_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "save_steps": 1,
        "save_total_limit": 3,
        "eval_steps": 2,
        "precision": "auto",
        "gradient_checkpointing": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # 1) train two optimizer steps
    run("scripts/train.py", "--config", str(config_path))
    assert (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "checkpoint-2").is_dir()
    meta = json.loads((adapter_dir / "training_metadata.json").read_text())
    assert meta["precision"] == "fp32" and meta["hardware"]["cuda_available"] is False
    assert meta["encoding"]["train"]["kept"] > 0 and meta["encoding"]["train"]["dropped_too_long"] == 0
    assert meta["dataset"]["dataset_hash"] == json.loads((prepared_dataset / "manifest.json").read_text())["dataset_hash"]
    assert meta["metrics"]["train_loss"] > 0 and "eval_loss" in meta["metrics"]
    assert meta["config"]["lora_r"] == 4 and meta["trainable_parameters"] > 0
    assert meta["resumed_from"] is None
    manifest = json.loads((adapter_dir / "artifact_manifest.json").read_text())
    assert manifest["kind"] == "lora_adapter" and manifest["dataset_hash"] == meta["dataset"]["dataset_hash"]
    assert all(not f.startswith("checkpoint-") for f in manifest["files"])

    # 2) resume from the last checkpoint for one more step
    run("scripts/train.py", "--config", str(config_path), "--max-steps", "3", "--resume")
    meta2 = json.loads((adapter_dir / "training_metadata.json").read_text())
    assert meta2["resumed_from"].endswith("checkpoint-2") and (adapter_dir / "checkpoint-3").is_dir()
    adapter_id = json.loads((adapter_dir / "artifact_manifest.json").read_text())["artifact_id"]

    # 3) merge
    merged_dir = tmp_path / "merged"
    run("scripts/merge_adapter.py", "--adapter-dir", str(adapter_dir), "--output-dir", str(merged_dir))
    merged = json.loads((merged_dir / "artifact_manifest.json").read_text())
    assert merged["kind"] == "merged_model" and merged["parent_artifact_id"] == adapter_id
    assert (merged_dir / "model.safetensors").exists() and not (merged_dir / "adapter_config.json").exists()

    # 4) evaluate merged ("finetuned") and base, then compare
    test_file = str(prepared_dataset / "test.jsonl")
    run(
        "-m",
        "qlora_vllm.evaluate",
        "run",
        "--backend",
        "hf",
        "--model",
        str(merged_dir),
        "--test",
        test_file,
        "--out",
        str(tmp_path / "ft"),
        "--variant",
        "finetuned",
        "--max-new-tokens",
        "4",
        "--batch-size",
        "4",
    )
    run(
        "-m",
        "qlora_vllm.evaluate",
        "run",
        "--backend",
        "hf",
        "--model",
        str(tiny_model_dir),
        "--test",
        test_file,
        "--out",
        str(tmp_path / "base"),
        "--variant",
        "base",
        "--max-new-tokens",
        "4",
        "--batch-size",
        "4",
    )
    ft = json.loads((tmp_path / "ft" / "results.json").read_text())
    base = json.loads((tmp_path / "base" / "results.json").read_text())
    assert ft["model"]["artifact_id"] == merged["artifact_id"] and ft["model"]["identity_verified"]
    assert base["model"]["kind"] == "hub_model"
    assert ft["dataset"]["sha256"] == base["dataset"]["sha256"] and ft["eval_config_hash"] == base["eval_config_hash"]
    assert ft["metrics"]["n"] == 6 and ft["metrics"]["error_rate"] == 0.0
    assert ft["timing"]["model_load_seconds"] is not None
    preds = [json.loads(line) for line in (tmp_path / "ft" / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 6 and {"id", "prediction", "references", "exact_match", "f1", "error"} <= preds[0].keys()
    run(
        "-m",
        "qlora_vllm.evaluate",
        "compare",
        "--report",
        f"base={tmp_path / 'base' / 'results.json'}",
        "--report",
        f"finetuned={tmp_path / 'ft' / 'results.json'}",
        "--out",
        str(tmp_path / "cmp"),
        "--resamples",
        "200",
    )
    cmp = json.loads((tmp_path / "cmp" / "comparison.json").read_text())
    assert "finetuned:base" in cmp["paired_bootstrap"] and (tmp_path / "cmp" / "comparison.md").exists()

    # 5) the evaluator refuses to compare reports from different test sets
    other = tmp_path / "other"
    other.mkdir()
    bad = dict(base, dataset=dict(base["dataset"], sha256="0" * 64))
    (other / "results.json").write_text(json.dumps(bad))
    (other / "predictions.jsonl").write_text((tmp_path / "base" / "predictions.jsonl").read_text())
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qlora_vllm.evaluate",
            "compare",
            "--report",
            f"a={other / 'results.json'}",
            "--report",
            f"b={tmp_path / 'ft' / 'results.json'}",
            "--out",
            str(tmp_path / "cmp2"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "not comparable" in proc.stderr
