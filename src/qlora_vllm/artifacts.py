"""Artifact identity: content hashes and manifests for models, adapters and datasets.

Every model artifact directory produced by this project carries an
``artifact_manifest.json`` whose ``artifact_id`` is a SHA-256 over its weight
and config files. Evaluation reports, benchmark reports and release decisions
reference that id, which is what lets the release gate detect reports that
describe a different checkpoint than the one being promoted.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "artifact_manifest.json"
# Files whose bytes define the identity of a model/adapter directory.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")
CONFIG_NAMES = ("config.json", "adapter_config.json", "generation_config.json", "quantization_config.json")
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "llmcompressor",
    "compressed-tensors",
    "vllm",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def identity_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and (p.suffix in WEIGHT_SUFFIXES or p.name in CONFIG_NAMES)
        # Trainer checkpoints and the rollback copies written by promote.py are not part of the identity.
        and not any(part.startswith(("checkpoint-", "previous-")) for part in p.relative_to(directory).parts)
    ]
    return sorted(files)


def hash_artifact_dir(directory: Path) -> tuple[str, dict[str, str]]:
    """Return ``(artifact_id, {relative_path: sha256})`` for a model directory."""
    directory = Path(directory)
    files = identity_files(directory)
    if not files:
        raise FileNotFoundError(f"no weight/config files found under {directory}")
    per_file = {str(p.relative_to(directory)): sha256_file(p) for p in files}
    combined = hashlib.sha256()
    for rel, digest in sorted(per_file.items()):
        combined.update(f"{rel}:{digest}\n".encode())
    return combined.hexdigest(), per_file


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def write_artifact_manifest(directory: Path, kind: str, **extra: Any) -> dict[str, Any]:
    """Hash ``directory`` and write its manifest. ``extra`` records provenance."""
    directory = Path(directory)
    artifact_id, files = hash_artifact_dir(directory)
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "artifact_id": artifact_id,
        "created_at": utc_now(),
        "files": files,
        "packages": package_versions(),
        **extra,
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_artifact_manifest(directory: Path) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; every model artifact must carry a manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifact_dir(directory: Path) -> dict[str, Any]:
    """Re-hash the directory and confirm it matches its manifest. Raises on mismatch."""
    manifest = read_artifact_manifest(directory)
    artifact_id, _ = hash_artifact_dir(directory)
    if artifact_id != manifest["artifact_id"]:
        raise ValueError(f"artifact {directory} has been modified: manifest id {manifest['artifact_id'][:12]}, current content id {artifact_id[:12]}")
    return manifest


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
