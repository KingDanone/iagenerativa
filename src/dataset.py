"""Dataset de language modeling autoregressivo.

Sequencia [t0,t1,t2,t3,...] -> input [t0,t1,t2], target [t1,t2,t3].
P(token seguinte | anteriores). Ver docs/MATH.md.

Formato processado: numpy uint16 (vocab < 65536) em shards:
  data/processed/train_000.npy, val_000.npy ... (ou train.bin/val.bin)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    def __init__(self, tokens: np.ndarray, block_size: int):
        assert tokens.ndim == 1, "tokens deve ser 1-D"
        assert len(tokens) > block_size + 1, "tokens insuficientes p/ block_size"
        # garante int64 p/ embedding; mantem uint16 em disco
        self.tokens = tokens
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return int(len(self.tokens) - self.block_size - 1)

    def __getitem__(self, i: int):
        x = self.tokens[i : i + self.block_size].astype(np.int64)
        y = self.tokens[i + 1 : i + 1 + self.block_size].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_token_arrays(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega shards train_*.npy e val_*.npy (fallback train.bin/val.bin)."""
    d = Path(data_dir)
    train_files = sorted(d.glob("train_*.npy")) or ([d / "train.bin"] if (d / "train.bin").exists() else [])
    val_files = sorted(d.glob("val_*.npy")) or ([d / "val.bin"] if (d / "val.bin").exists() else [])
    if not train_files:
        # compat: train.npy / val.npy
        if (d / "train.npy").exists():
            train_files = [d / "train.npy"]
        if (d / "val.npy").exists():
            val_files = [d / "val.npy"]
    if not train_files:
        raise FileNotFoundError(f"nenhum shard de treino em {d} (rode scripts/build_dataset.py)")
    # 1 shard (caso comum): usa mmap direto, sem copiar 1GB+ p/ RAM
    if len(train_files) == 1 and val_files:
        train = np.load(str(train_files[0]), mmap_mode="r")
        if str(train.dtype) != "uint16":
            train = train.astype(np.uint16)
        val = np.load(str(val_files[0]), mmap_mode="r")
        if str(val.dtype) != "uint16":
            val = val.astype(np.uint16)
        return train, val
    train = np.concatenate([np.load(str(f), mmap_mode="r").astype(np.uint16) for f in train_files])
    if val_files:
        val = np.concatenate([np.load(str(f), mmap_mode="r").astype(np.uint16) for f in val_files])
    else:  # fallback: 5% final como validacao
        n_val = max(1024, int(len(train) * 0.05))
        val, train = train[-n_val:], train[:-n_val]
    return train, val


def encode_and_shard(
    texts,
    tokenizer,
    out_dir: str | Path,
    shard_tokens: int = 1_000_000,
    val_frac: float = 0.05,
    add_eos: bool = True,
    shuffle_seed: int | None = 42,
) -> dict:
    """Tokeniza textos e salva shards uint16 train/val. Retorna estatisticas.

    Embaralha os docs (seed fixa) ANTES do split: sem isso a val seria so a
    cauda do corpus (ex. seed dialogica concentrada no fim) e a metrica viesa.
    """
    import random

    from src.tokenizer import BPETokenizer  # lazy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    n_docs = 0
    n_chars = 0
    eos = [tokenizer.eos_id] if add_eos else []
    if shuffle_seed is not None and not isinstance(texts, list):
        texts = list(texts)  # gerador -> lista p/ embaralhar
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(texts)
    for t in texts:
        if not t or not t.strip():
            continue
        n_docs += 1
        n_chars += len(t)
        buf.extend(tokenizer.encode(t) + eos)
    # uint16 exige vocab < 65536 e ids < 65536: ok
    toks = np.asarray(buf, dtype=np.uint16)
    n_val = max(256, int(len(toks) * val_frac))
    train_toks, val_toks = toks[:-n_val], toks[-n_val:]
    np.save(str(out / "train_000.npy"), train_toks)
    np.save(str(out / "val_000.npy"), val_toks)
    return {
        "docs": n_docs,
        "chars": n_chars,
        "train_tokens": int(len(train_toks)),
        "val_tokens": int(len(val_toks)),
        "total_tokens": int(len(toks)),
    }
