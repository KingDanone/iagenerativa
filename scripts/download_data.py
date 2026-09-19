#!/usr/bin/env python3
"""Download de dados PT com streaming e limite de tamanho.

Fontes suportadas (ver docs/DATASET_SELECTION.md + licencas em SOURCE_LICENSES):
  - fineweb2 : HuggingFaceFW/fineweb-2 (por_Latn) via streaming — web PT limpa/dedupada
  - wikipedia : HuggingFace wikimedia/wikipedia (20231101.pt) via streaming
  - carolina  : HuggingFace carolina-c4ai/corpus-carolina via streaming
  - oscar     : HuggingFace oscar-corpus/OSCAR-2201 (pt) via streaming
  - conversacional: QA-PT (Jpzinn654 + piaf) no formato do chat
  - aya       : CohereForAI/aya_dataset filtrado PT — instrucoes humanas
  - oasst     : OpenAssistant/oasst1 filtrado PT — dialogos multi-turno humanos
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
    p.add_argument("--sources", default="fineweb2,wikipedia,conversacional,aya,oasst,seed",
                   help="lista csv: fineweb2,wikipedia,carolina,oscar,conversacional,aya,oasst,seed")
    p.add_argument("--max-gb", type=float, default=2.0)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--output", default="data/raw/corpus.jsonl")
    p.add_argument("--lang", default="pt")
    return p.parse_args()


def budget_ok(path: Path, max_gb: float) -> bool:
    return path.exists() and path.stat().st_size >= max_gb * 1024**3


class JsonlWriter:
    """Escritor com handle persistente (antes: open/close por doc, 25k opens)."""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, source: str, text: str) -> int:
        if not text or len(text.strip()) < 50:
            return 0
        self.f.write(json.dumps({"source": source, "text": text}, ensure_ascii=False) + "\n")
        return len(text.encode("utf-8"))

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def append_jsonl(path: Path, source: str, text: str) -> int:
    """Compat: abre/fecha por chamada (lento; prefira JsonlWriter)."""
    w = JsonlWriter(path)
    try:
        return w.write(source, text)
    finally:
        w.close()


def fetch_wikipedia(w: JsonlWriter, max_bytes: int, max_docs: int | None) -> tuple[int, int]:
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
        n_bytes += w.write("wikipedia", f"{title}\n\n{txt}")
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
        if n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


def fetch_generic_hf(dataset: str, config: str | None, text_field: str, source: str, w: JsonlWriter,
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
        n_bytes += w.write(source, txt)
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
        if start_bytes + n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


def fetch_fineweb2_por(w: JsonlWriter, max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
    """FineWeb-2 por_Latn: web PT ja limpa/dedupada (datatrove), licenca ODC-By."""
    from datasets import load_dataset
    n_docs = n_bytes = 0
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True)
    except Exception as e:
        print(f"AVISO: fineweb-2 por_Latn indisponivel: {e}. Pulando.")
        return 0, 0
    for row in ds:
        try:
            txt = str(row.get("text") or "").strip()
        except Exception:
            continue
        if len(txt) < 300:
            continue
        n_bytes += w.write("fineweb2", txt)
        n_docs += 1
        if n_docs % 20000 == 0:
            print(f"  fineweb2: docs={n_docs} mb={n_bytes/1e6:.0f}", flush=True)
        if max_docs and n_docs >= max_docs:
            break
        if start_bytes + n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


_PT_LANGS = {"pt", "por", "por_latn", "portuguese", "portugues", "português", "pt-br", "pt_br"}


def _row_lang(row: dict) -> str:
    for k in ("language_code", "language", "lang", "locale"):
        v = row.get(k)
        if v:
            return str(v).strip().lower()
    return ""


def _row_field(row: dict, names: tuple[str, ...]) -> str:
    for k in names:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def fetch_aya_pt(w: JsonlWriter, max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
    """CohereForAI/aya_dataset filtrado p/ PT: instrucoes humanas (Apache-2.0)."""
    from datasets import load_dataset
    n_docs = n_bytes = 0
    try:
        ds = load_dataset("CohereForAI/aya_dataset", split="train", streaming=True)
    except Exception as e:
        print(f"AVISO: aya_dataset indisponivel: {e}. Pulando.")
        return 0, 0
    for row in ds:
        try:
            if _row_lang(row) not in _PT_LANGS:
                continue
            q = _row_field(row, ("inputs", "input", "instruction", "prompt", "question"))
            a = _row_field(row, ("targets", "target", "output", "response", "answer"))
            if not q or not a:
                continue
            txt = f"### Usuário:\n{q}\n\n### Assistente:\n{a}".strip()
            if len(txt) < 80:
                continue
            n_bytes += w.write("aya_pt", txt)
            n_docs += 1
            if max_docs and n_docs >= max_docs:
                break
            if start_bytes + n_bytes >= max_bytes:
                break
        except Exception:
            continue
    return n_docs, n_bytes


def _serialize_oasst_thread(msgs: list[dict]) -> str | None:
    """Monta blocos ### Usuário/### Assistente a partir de msgs (id->msg, parent_id)."""
    by_id = {m["id"]: m for m in msgs}
    children: dict[str, list[str]] = {}
    roots = []
    for m in msgs:
        p = m.get("parent")
        if p and p in by_id:
            children.setdefault(p, []).append(m["id"])
        else:
            roots.append(m["id"])
    # segue a cadeia mais longa a partir de cada raiz prompter
    best: list[dict] = []
    for r in roots:
        chain = [by_id[r]]
        cur = r
        while cur in children:
            nxt = sorted(children[cur], key=lambda i: by_id[i].get("rank", 0) or 0)[0]
            chain.append(by_id[nxt])
            cur = nxt
        if len(chain) > len(best):
            best = chain
    blocks = []
    i = 0
    while i + 1 < len(best):
        u, a = best[i], best[i + 1]
        if u.get("role") == "prompter" and a.get("role") == "assistant" and u.get("text") and a.get("text"):
            blocks.append(f"### Usuário:\n{u['text'].strip()}\n\n### Assistente:\n{a['text'].strip()}")
            i += 2
        else:
            i += 1
    if not blocks:
        return None
    return "\n\n".join(blocks).strip()


def fetch_oasst_pt(w: JsonlWriter, max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
    """OpenAssistant/oasst1 filtrado lang==pt: dialogos multi-turno humanos (Apache-2.0).

    Bufferiza so mensagens PT (subconjunto pequeno) e monta threads por
    message_tree_id ao final — streaming puro mistura arvores.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    except Exception as e:
        print(f"AVISO: oasst1 indisponivel: {e}. Pulando.")
        return 0, 0
    trees: dict[str, list[dict]] = {}
    for row in ds:
        try:
            lang = str(row.get("lang") or "").strip().lower()
            if lang not in _PT_LANGS:
                continue
            tid = str(row.get("message_tree_id") or row.get("message_id") or "")
            txt = str(row.get("text") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            if not tid or not txt or role not in ("prompter", "assistant"):
                continue
            trees.setdefault(tid, []).append({
                "id": str(row.get("message_id") or f"{tid}#{len(trees.get(tid, []))}"),
                "parent": str(row.get("parent_id") or ""),
                "text": txt, "role": role, "rank": row.get("rank"),
            })
        except Exception:
            continue
    print(f"  oasst: {len(trees)} threads PT encontradas", flush=True)
    n_docs = n_bytes = 0
    for msgs in trees.values():
        txt = _serialize_oasst_thread(msgs)
        if not txt or len(txt) < 80:
            continue
        n_bytes += w.write("oasst_pt", txt)
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
        if start_bytes + n_bytes >= max_bytes:
            break
    return n_docs, n_bytes


def fetch_conversacional(w: JsonlWriter, max_bytes: int, max_docs: int | None, start_bytes: int) -> tuple[int, int]:
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
            got = 0
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
                    total_bytes += w.write("conversacional", txt)
                    total_docs += 1
                    got += 1
                    if max_docs and total_docs >= max_docs:
                        break
                    if start_bytes + total_bytes >= max_bytes:
                        break
                except Exception:
                    continue
            if got > 0:
                break  # so tenta a proxima fonte se esta rendeu 0 docs
    except ImportError:
        print("datasets lib ausente p/ conversacional HF; usando seed local.")
    return total_docs, total_bytes


def copy_seed(w: JsonlWriter) -> tuple[int, int]:
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
        b += w.write("seed_conversacional", ch)
        docs += 1
    return docs, b


SOURCE_LICENSES = {
    "fineweb2": "ODC-By 1.0 (HuggingFaceFW/fineweb-2, por_Latn) — atribuicao obrigatoria",
    "wikipedia": "CC BY-SA (wikimedia/wikipedia 20231101.pt)",
    "carolina": "ver carolina-c4ai/corpus-carolina (mista; checar por doc)",
    "oscar": "ver oscar-corpus/OSCAR-2201 (mista; checar por doc)",
    "conversacional": "Jpzinn654/qa-portuguese-small (verificar) / piaf",
    "aya": "Apache-2.0 (CohereForAI/aya_dataset, subset PT)",
    "oasst": "Apache-2.0 (OpenAssistant/oasst1, subset PT)",
    "seed_conversacional": "propria, livre",
}


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # download reproduzivel do zero
    max_bytes = int(args.max_gb * 1024**3)
    # divisao de orcamento v2: base web PT + formal + conversa (soma = 100%)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    grand_docs = grand_bytes = 0
    per_source_cap = {
        "fineweb2": int(max_bytes * 0.50),
        "wikipedia": int(max_bytes * 0.15),
        "carolina": int(max_bytes * 0.10),
        "oscar": int(max_bytes * 0.05),
        "conversacional": int(max_bytes * 0.08),
        "aya": int(max_bytes * 0.07),
        "oasst": int(max_bytes * 0.04),
        "seed": int(max_bytes * 0.01) + 1_000_000,
    }
    w = JsonlWriter(out)
    try:
        for src in sources:
            remaining = max_bytes - grand_bytes
            if remaining <= 0:
                break
            cap = min(per_source_cap.get(src, remaining), remaining)
            if src == "fineweb2":
                d, b = fetch_fineweb2_por(w, cap, args.max_docs, grand_bytes)
            elif src == "wikipedia":
                d, b = fetch_wikipedia(w, cap, args.max_docs)
            elif src == "carolina":
                d, b = fetch_generic_hf("carolina-c4ai/corpus-carolina", None, "text", "carolina",
                                        w, cap, args.max_docs, grand_bytes)
            elif src == "oscar":
                d, b = fetch_generic_hf("oscar-corpus/OSCAR-2201", "pt", "text", "oscar",
                                        w, cap, args.max_docs, grand_bytes)
            elif src == "conversacional":
                d, b = fetch_conversacional(w, cap, args.max_docs, grand_bytes)
            elif src == "aya":
                d, b = fetch_aya_pt(w, cap, args.max_docs, grand_bytes)
            elif src == "oasst":
                d, b = fetch_oasst_pt(w, cap, args.max_docs, grand_bytes)
            elif src == "seed":
                d, b = copy_seed(w)
            else:
                print(f"fonte desconhecida: {src}")
                continue
            grand_docs += d
            grand_bytes += b
            print(f"fonte={src} docs={d} bytes={b/1e6:.1f}MB acumulado={grand_bytes/1e6:.1f}MB")
    finally:
        w.close()
    print(f"OK: {grand_docs} docs, {grand_bytes/1024**2:.1f} MB -> {out}")
    print("licencas por fonte (reproduzir em docs/DATASET_SELECTION.md):")
    for src in sources:
        print(f"  {src}: {SOURCE_LICENSES.get(src, '?')}")


if __name__ == "__main__":
    main()
