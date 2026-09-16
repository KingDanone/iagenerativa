#!/usr/bin/env python3
"""Inspeciona corpus.jsonl: conta docs, bytes, chars, palavras, distribuicao por fonte."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus.jsonl")
    p.add_argument("--sample", type=int, default=5)
    args = p.parse_args()
    path = Path(args.input)
    n = 0
    chars = words = 0
    by_src: Counter = Counter()
    small = 0
    for i, line in enumerate(open(path, encoding="utf-8")):
        try:
            row = json.loads(line)
        except Exception:
            continue
        t = row.get("text", "")
        n += 1
        chars += len(t)
        words += len(t.split())
        by_src[row.get("source", "?")] += 1
        if len(t) < 100:
            small += 1
        if i < args.sample:
            print(f"--- doc {i} src={row.get('source')} chars={len(t)} ---")
            print(t[:400].replace("\n", " ") + "...\n")
    print(f"docs={n} chars={chars:,} palavras={words:,} bytes~{chars:,} | por fonte={dict(by_src)} | curtos(<100ch)={small}")


if __name__ == "__main__":
    main()
