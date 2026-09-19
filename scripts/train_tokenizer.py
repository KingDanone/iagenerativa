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


def iter_texts(path: Path, max_chars: int = 60_000_000):
    """Amostra estratificada (stride) em 2 passadas, O(1) RAM.

    Passada 1: conta docs/chars. Passada 2: emite 1 doc a cada `stride`,
    cobrindo inicio/meio/fim de todas as fontes ate o orcamento.
    """
    import json
    n_docs = total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t.strip():
                n_docs += 1
                total += len(t)
    if n_docs == 0:
        return
    stride = max(1, total // max(1, max_chars))
    budget = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride != 0:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t.strip():
                yield t
                budget += len(t)
                if budget >= max_chars:
                    break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus.clean.jsonl")
    p.add_argument("--output", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--max-chars", type=int, default=60_000_000)
    p.add_argument("--max-words", type=int, default=300_000)
    args = p.parse_args()
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.train(iter_texts(Path(args.input), args.max_chars),
              vocab_size=args.vocab_size, max_words=args.max_words)
    tok.save(args.output)
    print(f"tokenizer salvo: {args.output} vocab={len(tok)} merges={len(tok.merges)}")
    for s in ["Olá, mundo!", "São Luís é uma cidade brasileira.", "Inteligência artificial",
              "ação", "informação", "programação", "coração"]:
        ids = tok.encode(s)
        print(f"{s!r} -> {ids[:16]} -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
