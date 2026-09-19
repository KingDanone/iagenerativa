#!/usr/bin/env python3
"""Constroi dataset tokenizado: python scripts/build_dataset.py --input data/raw/corpus.clean.jsonl"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import encode_and_shard  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402


def iter_texts(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t.strip():
                yield t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus.clean.jsonl")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="seed do embaralhamento de docs antes do split (0/-1 = sem shuffle)")
    args = p.parse_args()
    tok = BPETokenizer.load(args.tokenizer)
    seed = None if args.shuffle_seed is not None and args.shuffle_seed < 0 else args.shuffle_seed
    stats = encode_and_shard(iter_texts(Path(args.input)), tok, args.out_dir,
                             val_frac=args.val_frac, shuffle_seed=seed)
    print(f"dataset ok: {stats}")
    # registra contagem de palavras aproximada
    print(f"train_tokens={stats['train_tokens']:,} val_tokens={stats['val_tokens']:,}")


if __name__ == "__main__":
    main()
