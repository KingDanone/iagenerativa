#!/usr/bin/env python3
"""Prepara manifesto reproduzivel: metadata/dataset_manifest.json (fontes, datas, tamanhos, filtros)."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", default="data/raw/corpus.clean.jsonl")
    p.add_argument("--raw", default="data/raw/corpus.jsonl")
    p.add_argument("--out", default="metadata/dataset_manifest.json")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    from collections import Counter
    by_src: Counter = Counter()
    n = chars = 0
    cp = Path(args.clean)
    if cp.exists():
        for line in open(cp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            chars += len(r.get("text", ""))
            by_src[r.get("source", "?")] += 1
    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw_file": args.raw,
        "clean_file": args.clean,
        "raw_sha256": sha(Path(args.raw)) if Path(args.raw).exists() else None,
        "clean_sha256": sha(cp) if cp.exists() else None,
        "docs": n,
        "chars": chars,
        "by_source": dict(by_src),
        "licenses": {
            "fineweb2": "ODC-By 1.0 (HuggingFaceFW/fineweb-2 por_Latn)",
            "wikipedia": "CC BY-SA (wikimedia/wikipedia 20231101.pt)",
            "carolina": "mista (ver corpus-carolina)",
            "oscar": "mista (ver OSCAR-2201)",
            "conversacional": "Jpzinn654/qa-portuguese-small / piaf (verificar)",
            "aya_pt": "Apache-2.0 (CohereForAI/aya_dataset)",
            "oasst_pt": "Apache-2.0 (OpenAssistant/oasst1)",
            "seed_conversacional": "propria, livre",
        },
        "filters": {"min_chars": 100, "max_chars": 100000, "dedup": "normalized sha256",
                    "lang_filter": "stopwords+acentos PT", "unicode": "NFC",
                    "spam_blocklist": "catracalivre/promo/bets"},
        "notes": args.notes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
