"""Hardware detection and precision selection.

A T4 (compute capability 7.5) has no native BF16 support, so the trainer must
not assume BF16. ``resolve_precision`` turns the config's ``precision`` setting
(``auto``, ``fp16``, ``bf16`` or ``fp32``) into a concrete torch dtype.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": _cpu_count(),
        "cuda_available": False,
        "gpus": [],
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency for training
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return info


def bf16_supported(hardware: dict[str, Any]) -> bool:
    return bool(hardware.get("cuda_available")) and all(g["bf16_supported"] for g in hardware["gpus"])


def resolve_precision(setting: str, hardware: dict[str, Any] | None = None) -> str:
    """Return one of ``fp16``, ``bf16`` or ``fp32``.

    ``auto`` picks BF16 when every visible GPU supports it, FP16 on other GPUs
    (the T4 case) and FP32 on CPU. An explicit ``bf16`` request on hardware
    without BF16 support is rejected instead of silently degrading.
    """
    hardware = hardware or detect_hardware()
    setting = (setting or "auto").lower()
    if setting == "auto":
        if not hardware.get("cuda_available"):
            return "fp32"
        return "bf16" if bf16_supported(hardware) else "fp16"
    if setting == "bf16" and not bf16_supported(hardware):
        raise RuntimeError("precision=bf16 requested but the detected hardware does not support BF16; use fp16 or auto")
    if setting not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unknown precision setting: {setting}")
    return setting


def torch_dtype(precision: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _nvidia_smi() -> dict[str, Any] | None:
    """GPU memory as reported by nvidia-smi, or ``None`` when it is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, used, total = [part.strip() for part in line.split(",")]
        gpus.append({"name": name, "memory_used_mib": int(used), "memory_total_mib": int(total)})
    return {"gpus": gpus}
