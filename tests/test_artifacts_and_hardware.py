from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_vllm.artifacts import hash_artifact_dir, read_artifact_manifest, verify_artifact_dir, write_artifact_manifest
from qlora_vllm.hardware import resolve_precision

CPU = {"cuda_available": False, "gpus": []}
T4 = {"cuda_available": True, "gpus": [{"name": "Tesla T4", "compute_capability": "7.5", "bf16_supported": False}]}
A100 = {"cuda_available": True, "gpus": [{"name": "A100", "compute_capability": "8.0", "bf16_supported": True}]}


def test_artifact_id_covers_weights_and_configs_but_not_checkpoints(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "README.md").write_text("docs do not affect identity")
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-5" / "adapter_model.safetensors").write_bytes(b"ckpt")
    artifact_id, files = hash_artifact_dir(tmp_path)
    assert set(files) == {"config.json", "model.safetensors"}
    (tmp_path / "README.md").write_text("changed")
    assert hash_artifact_dir(tmp_path)[0] == artifact_id
    (tmp_path / "model.safetensors").write_bytes(b"weights2")
    assert hash_artifact_dir(tmp_path)[0] != artifact_id


def test_manifest_roundtrip_and_tamper_detection(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = write_artifact_manifest(tmp_path, kind="merged_model", base_model="x", parent_artifact_id="p")
    assert read_artifact_manifest(tmp_path) == manifest
    assert verify_artifact_dir(tmp_path)["artifact_id"] == manifest["artifact_id"]
    assert manifest["packages"]["transformers"] and manifest["created_at"].endswith("+00:00")
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified"):
        verify_artifact_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_artifact_manifest(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        hash_artifact_dir(tmp_path / "missing")


def test_precision_resolution_is_hardware_aware():
    assert resolve_precision("auto", CPU) == "fp32"
    assert resolve_precision("auto", T4) == "fp16"
    assert resolve_precision("auto", A100) == "bf16"
    assert resolve_precision("fp16", T4) == "fp16"
    with pytest.raises(RuntimeError, match="BF16"):
        resolve_precision("bf16", T4)
    with pytest.raises(ValueError):
        resolve_precision("int8", A100)


def test_legacy_config_keys_map_to_precision(tmp_path: Path):
    from qlora_vllm.config import load_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text(json.dumps({"model_name": "m", "dataset_path": "d", "output_dir": "o", "bf16": False, "fp16": True}))
    assert load_config(cfg).precision == "fp16"
    cfg.write_text(json.dumps({"model_name": "m", "dataset_path": "d", "output_dir": "o", "unknown_key": 1}))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(cfg)
