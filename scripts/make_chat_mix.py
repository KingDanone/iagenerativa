#!/usr/bin/env python3
"""Monta mix conversacional p/ ajuste curto: toda a conversa + replay de conhecimento.

Uso: .venv/bin/python scripts/make_chat_mix.py --input data/raw/corpus_v2.clean.jsonl \
         --output data/raw/corpus_chat.jsonl --replay-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CONV = {"conversacional", "aya_pt", "oasst_pt", "seed_conversacional"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus_v2.clean.jsonl")
    p.add_argument("--output", default="data/raw/corpus_chat.jsonl")
    p.add_argument("--replay-frac", type=float, default=0.2,
                   help="fração de docs de conhecimento vs. conversa (evita esquecimento)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    conv: list[dict] = []
    pool: list[dict] = []  # conhecimento candidato a replay
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            (conv if r.get("source") in CONV else pool).append(r)
    n_replay = int(len(conv) * args.replay_frac)
    replay = rng.sample(pool, min(n_replay, len(pool)))
    mix = conv + replay
    rng.shuffle(mix)
    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for r in mix:
            f.write(json.dumps({"source": r["source"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"conv={len(conv)} replay={len(replay)} total={len(mix)} -> {out}")


if __name__ == "__main__":
    main()
