"""Utilidades: seed, device, GPU, checkpoints, LR schedule."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda solicitado mas CUDA nao disponivel")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"device desconhecido: {name!r} (use auto|cuda|cpu)")


def gpu_info() -> dict:
    info: dict = {"cuda_available": torch.cuda.is_available()}
    try:
        info["torch_version"] = torch.__version__
        info["cuda_runtime"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "device_index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated(idx) / 1024**3, 3),
                }
            )
        except Exception as e:  # pragma: no cover
            info["error"] = str(e)
    return info


def vram_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    i = torch.cuda.current_device()
    return {
        "cuda": True,
        "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1024**3, 3),
    }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M ({n:,})".replace(",", ".")
    if n >= 1_000:
        return f"{n/1_000:.1f}k ({n:,})".replace(",", ".")
    return str(n)


def lr_schedule(step: int, *, max_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Warmup linear + decaimento cosseno. Ver docs/TRAINING.md."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | os.PathLike, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike, map_location=None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
