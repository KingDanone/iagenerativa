#!/usr/bin/env python3
"""Download de dados PT com streaming e limite de tamanho.

Fontes suportadas (ver docs/DATASET_SELECTION.md):
  - wikipedia : HuggingFace wikimedia/wikipedia (20231101.pt) via streaming
  - carolina  : HuggingFace carolina-c4ai/corpus-carolina via streaming
  - oscar     : HuggingFace oscar-corpus/OSCAR-2201 (pt) via streaming
  - conversacional: SQuAD-PT (piaf/quac?) + QA-PT + seed local
  - seed      : data/seed_conversational.txt (sempre incluido, pequeno)

Uso:
  python scripts/download_data.py --sources wikipedia,conversacional --max-gb 2
  python scripts/download_data.py --sources wikipedia --max-docs 2000 --output data/raw/wiki.jsonl
Sempre processa em fluxo (nao carrega GBs na RAM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="wikipedia,conversacional,seed",
                   help="lista csv: wikipedia,carolina,oscar,conversacional,seed")
    p.add_argument("--max-gb", type=float, default=2.0)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--output", default="data/raw/corpus.jsonl")
    p.add_argument("--lang", default="pt")
    return p.parse_args()


def budget_ok(path: Path, max_gb: float) -> bool:
    return path.exists() and path.stat().st_size >= max_gb * 1024**3


def append_jsonl(path: Path, source: str, text: str) -> int:
    if not text or len(text.strip()) < 50:
        return 0
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"source": source, "text": text}, ensure_ascii=False) + "\n")
    return len(text.encode("utf-8"))


def fetch_wikipedia(out: Path, max_bytes: int, max_docs: int | None) -> tuple[int, int]:
    from datasets import load_dataset
    n_docs = n_bytes = 0
    for cfg in ("20231101.pt",):
        try:
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            break
        except Exception as e:
            print(f"wikipedia {cfg} falhou: {e}")
            ds = None
    if ds is None:
        print("AVISO: wikipedia indisponivel (offline?). Pulando.")
        return 0, 0
    for row in ds:
        txt = (row.get("text") or "").strip()
        title = row.get("title") or ""
        if len(txt) < 300:
            continue
        n_bytes += append_jsonl(out, "wikipedia", f"{title}\n\n{txt}")
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
        if n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


def fetch_generic_hf(dataset: str, config: str | None, text_field: str, source: str, out: Path,
                     max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
    from datasets import load_dataset
    n_docs = n_bytes = 0
    try:
        kw = dict(split="train", streaming=True)
        ds = load_dataset(dataset, config, **kw) if config else load_dataset(dataset, **kw)
    except Exception as e:
        print(f"AVISO: {dataset} indisponivel: {e}. Pulando.")
        return 0, 0
    for row in ds:
        txt = str(row.get(text_field) or "").strip()
        if len(txt) < 200:
            continue
        n_bytes += append_jsonl(out, source, txt)
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
        if start_bytes + n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


def fetch_conversacional(out: Path, max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
    """QA/instrucao PT: tenta HF, senao usa seed local expandido."""
    total_docs = total_bytes = 0
    # 1) tenta dataset QA portugues pequeno
    tried = [
        ("Jpzinn654/qa-portuguese-small", None, ("question", "answer")),
        ("piaf", None, None),
    ]
    try:
        from datasets import load_dataset
        for ds_name, cfg, fields in tried:
            try:
                ds = load_dataset(ds_name, cfg, split="train", streaming=True) if cfg else load_dataset(ds_name, split="train", streaming=True)
            except Exception as e:
                print(f"conversacional {ds_name}: {e}")
                continue
            for row in ds:
                try:
                    if ds_name.startswith("Jpzinn654"):
                        q = ((row.get("question_title") or "") + " " + (row.get("question_text") or "")).strip()
                        a = (row.get("answer_text") or "").strip()
                    elif fields:
                        q, a = row.get(fields[0], ""), row.get(fields[1], "")
                    else:
                        q, a = row.get("question", ""), row.get("answers", "")
                        if isinstance(a, dict):
                            a = (a.get("text") or [""])[0]
                    if not q or q == "-":
                        q = row.get("question_title") or q
                    if not a or a == "-":
                        continue
                    txt = f"### Usuário:\n{q}\n\n### Assistente:\n{a}".strip()
                    if len(txt) < 80:
                        continue
                    total_bytes += append_jsonl(out, "conversacional", txt)
                    total_docs += 1
                    if max_docs and total_docs >= max_docs:
                        break
                    if start_bytes + total_bytes >= max_bytes:
                        break
                except Exception:
                    continue
            break
    except ImportError:
        print("datasets lib ausente p/ conversacional HF; usando seed local.")
    return total_docs, total_bytes


def copy_seed(out: Path) -> tuple[int, int]:
    import re
    seed = Path("data/seed_conversational.txt")
    if not seed.exists():
        return 0, 0
    raw = seed.read_text(encoding="utf-8")
    # cada exemplo comeca em "### Usuário:" e contem o par user+assistant
    chunks = [c.strip() for c in re.split(r"(?=### Usuário:)", raw) if c.strip()]
    docs = b = 0
    for ch in chunks:
        if "### Assistente:" not in ch or len(ch) < 40:
            continue
        b += append_jsonl(out, "seed_conversacional", ch)
        docs += 1
    return docs, b


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # download reproduzivel do zero
    max_bytes = int(args.max_gb * 1024**3)
    # divisao de orcamento: 55% geral, 25% adicional, 20% conversacional (ref. #10)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    grand_docs = grand_bytes = 0
    per_source_cap = {
        "wikipedia": int(max_bytes * 0.55),
        "carolina": int(max_bytes * 0.55),
        "oscar": int(max_bytes * 0.50),
        "conversacional": int(max_bytes * 0.20),
        "seed": int(max_bytes * 0.01) + 1_000_000,
    }
    for src in sources:
        remaining = max_bytes - grand_bytes
        if remaining <= 0:
            break
        cap = min(per_source_cap.get(src, remaining), remaining)
        if src == "wikipedia":
            d, b = fetch_wikipedia(out, cap, args.max_docs)
        elif src == "carolina":
            d, b = fetch_generic_hf("carolina-c4ai/corpus-carolina", None, "text", "carolina",
                                    out, cap, args.max_docs, 0)
        elif src == "oscar":
            d, b = fetch_generic_hf("oscar-corpus/OSCAR-2201", "pt", "text", "oscar",
                                    out, cap, args.max_docs, 0)
        elif src == "conversacional":
            d, b = fetch_conversacional(out, cap, args.max_docs, 0)
        elif src == "seed":
            d, b = copy_seed(out)
        else:
            print(f"fonte desconhecida: {src}");
            continue
        grand_docs += d
        grand_bytes += b
        print(f"fonte={src} docs={d} bytes={b/1e6:.1f}MB acumulado={grand_bytes/1e6:.1f}MB")
    print(f"OK: {grand_docs} docs, {grand_bytes/1024**2:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
