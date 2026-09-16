#!/usr/bin/env python3
"""Treina o tokenizer BPE: python scripts/train_tokenizer.py --input data/raw/corpus.clean.jsonl --vocab-size 4096"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tokenizer import BPETokenizer  # noqa: E402


def iter_texts(path: Path, max_chars: int = 12_000_000):
    """Amostra estratificada (stride) p/ cobrir todas as fontes sem ler 70MB."""
    import json
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t.strip():
                docs.append(t)
    # stride: cobre inicio/meio/fim de todas as fontes
    total = sum(len(d) for d in docs)
    stride = max(1, total // max_chars // 1) if total > max_chars else 1
    budget = 0
    for i in range(0, len(docs), int(stride)):
        t = docs[i]
        yield t
        budget += len(t)
        if budget >= max_chars:
            break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus.clean.jsonl")
    p.add_argument("--output", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=4096)
    args = p.parse_args()
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.train(iter_texts(Path(args.input)), vocab_size=args.vocab_size)
    tok.save(args.output)
    print(f"tokenizer salvo: {args.output} vocab={len(tok)} merges={len(tok.merges)}")
    for s in ["Olá, mundo!", "São Luís é uma cidade brasileira.", "Inteligência artificial",
              "ação", "informação", "programação", "coração"]:
        ids = tok.encode(s)
        print(f"{s!r} -> {ids[:16]} -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
