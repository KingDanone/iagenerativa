#!/usr/bin/env python3
"""Avaliacao: roda eval/prompts_ptbr.txt e salva artifacts/evaluation/"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.generation import generate, repetition_stats  # noqa: E402
from src.model import build_model  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.utils import ensure_dir, load_checkpoint, resolve_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--prompts", default="eval/prompts_ptbr.txt")
    p.add_argument("--out-dir", default="artifacts/evaluation")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def logits_finite(model, tok, prompt: str, device) -> bool:
    """Forward real no prompt: False se logits contiverem NaN/Inf (divergencia)."""
    try:
        ids = tok.encode(prompt)
        ctx = getattr(model, "block_size", 256)
        ids = ids[-ctx:] if len(ids) > ctx else ids
        if not ids:
            return True
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model.to(device).eval()(x)
        return bool(torch.isfinite(logits).all().item())
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    tok = BPETokenizer.load(args.tokenizer)
    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.strip().startswith("#")]
    out_dir = ensure_dir(args.out_dir)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    results = []
    for i, pr in enumerate(prompts):
        out = generate(model, tok, pr, args.max_new_tokens, args.temperature, args.top_k, args.top_p, device)
        rec = {"prompt": pr, "output": out, "model": "MiniGPT",
               "checkpoint": args.checkpoint, "temperature": args.temperature,
               "top_k": args.top_k, "top_p": args.top_p, "timestamp": ts,
               "repetition": repetition_stats(out),
               "empty": len(out.strip()) == 0,
               "has_nan": not logits_finite(model, tok, pr, device)}
        results.append(rec)
        print(f"[{i+1}/{len(prompts)}] P: {pr}\n   R: {out[:220]}\n")
    (out_dir / f"eval_{ts.replace(':','-')}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_empty = sum(r["empty"] for r in results)
    print(f"salvo em {out_dir} | prompts={len(results)} vazios={n_empty}")


if __name__ == "__main__":
    main()
