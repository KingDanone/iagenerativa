#!/usr/bin/env python3
"""Limpeza: unicode, espacos, HTML, lixo, vazios, dedup por hash, filtro PT basico.

Preserva acentos/pontuacao/maiusculas/paragrafos. Ver docs/TRAINING.md.
Uso: python scripts/clean_data.py --input data/raw/corpus.jsonl --output data/raw/corpus.clean.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"[ \t]+")
PT_STOP = {" de ", " que ", " para ", " com ", " uma ", " dos ", " nas ", " pelo ", " sobre ", " entre "}

# spam recorrente no QA-PT (dominios nus que o URL_RE nao pega + promocoes)
SPAM_RES = [
    re.compile(r"(?i)catracalivre\b"),
    re.compile(r"(?i)(c[oó]digo\s+promocional|cupom\s+de\s+desconto|frete\s+gr[aá]tis|clique\s+aqui|ganhe\s+dinheiro|aposte\s+agora|renda\s+extra\s+em\s+casa)"),
    re.compile(r"(?i)(cassino|bet365|blaze\s+apostas|jogo\s+do\s+tigrinho)"),
]


def is_spam(t: str) -> bool:
    return any(r.search(t) for r in SPAM_RES)


def clean_text(t: str) -> str:
    t = html.unescape(t)
    t = HTML_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in t.split("\n"):
        ln = WS_RE.sub(" ", ln).strip(" \t")
        lines.append(ln)
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def looks_portuguese(t: str) -> bool:
    low = " " + t.lower() + " "
    hits = sum(1 for s in PT_STOP if s in low)
    has_accent = any(c in t for c in "ãõçáéíóúâêôàüÁÉÍÓÚÃÕÇ")
    return hits >= 1 or (has_accent and len(t.split()) >= 8)


def norm_hash(t: str) -> str:
    n = re.sub(r"\s+", " ", t.strip().lower())
    return hashlib.sha256(n.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/corpus.jsonl")
    p.add_argument("--output", default="data/raw/corpus.clean.jsonl")
    p.add_argument("--min-chars", type=int, default=100)
    p.add_argument("--max-chars", type=int, default=100000)
    p.add_argument("--filter-pt", action="store_true", default=True)
    p.add_argument("--no-filter-pt", dest="filter_pt", action="store_false")
    p.add_argument("--block-spam", action="store_true", default=True)
    p.add_argument("--no-block-spam", dest="block_spam", action="store_false")
    args = p.parse_args()
    seen: set[str] = set()
    kept = dropped = dup = spam = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except Exception:
                dropped += 1
                continue
            t = clean_text(row.get("text", ""))
            if not t or len(t) < args.min_chars or len(t) > args.max_chars:
                dropped += 1
                continue
            if args.block_spam and is_spam(t):
                spam += 1
                continue
            if args.filter_pt and not looks_portuguese(t):
                dropped += 1
                continue
            h = norm_hash(t)
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            fout.write(json.dumps({"source": row.get("source", "?"), "text": t}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} dropped={dropped} spam={spam} dups={dup} -> {args.output}")


if __name__ == "__main__":
    main()
