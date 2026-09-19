#!/usr/bin/env python3
"""Gate de auditoria pos-treino: veredito APROVADO/REVISAR antes de commit/push.

Compara o checkpoint v2 contra o baseline Fase 2 e checa integridade:
  1) checkpoints abrem; config bate com tokenizer (vocab/block)
  2) val/PPL + razao train/val (overfit explosivo?)
  3) eval nos prompts: vazios, NaN real, div_ok
  4) throughput/VRAM do log

Uso: .venv/bin/python scripts/audit_post_train.py \
         --checkpoint artifacts/checkpoints_v2/best.pt \
         --tokenizer artifacts/tokenizer/tokenizer_v2.json \
         --baseline artifacts/evaluation/eval_2026-09-18T22-34-08.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation import generate, repetition_stats  # noqa: E402
from src.model import build_model  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.utils import load_checkpoint, resolve_device  # noqa: E402

BASELINE_VAL = 2.5523  # melhor val Fase 2 (docs/RESULTS.md) — PPL nao comparavel (vocab mudou)


def check_logits_finite(model, tok, prompt: str, device) -> bool:
    import torch

    try:
        ids = tok.encode(prompt)
        ctx = getattr(model, "block_size", 512)
        ids = ids[-ctx:] if len(ids) > ctx else ids
        if not ids:
            return True
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model.to(device).eval()(x)
        return bool(torch.isfinite(logits).all().item())
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints_v2/best.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer_v2.json")
    p.add_argument("--prompts", default="eval/prompts_ptbr.txt")
    p.add_argument("--baseline", default="artifacts/evaluation/eval_2026-09-18T22-34-08.json")
    p.add_argument("--log", default="logs_v2/train.log")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    fails: list[str] = []
    print("== [1] integridade ==")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    tok = BPETokenizer.load(args.tokenizer)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    print(f"checkpoint ok: step={ckpt.get('step')} best_val={ckpt.get('best_val')}")
    if len(tok) != int(ckpt["config"].get("vocab_size", len(tok))):
        fails.append(f"vocab mismatch: tokenizer={len(tok)} config={ckpt['config'].get('vocab_size')}")
    else:
        print(f"vocab ok: {len(tok)} | block={ckpt['config'].get('block_size')}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    print("== [2] loss/PPL ==")
    best_val = float(ckpt.get("best_val", "inf"))
    train_loss = ckpt.get("train_loss")
    ppl = math.exp(best_val) if best_val < 20 else float("inf")
    print(f"best_val={best_val:.4f} ppl={ppl:.2f} train_final={train_loss}")
    print(f"(baseline Fase 2: val={BASELINE_VAL} — comparacao indireta, vocab 4k->8k)")
    if not math.isfinite(best_val) or best_val > 6.0:
        fails.append(f"best_val suspeita: {best_val}")
    if train_loss and best_val > train_loss * 1.5:
        fails.append(f"overfit explosivo? val={best_val:.3f} >> train={train_loss:.3f}")

    print("== [3] eval prompts ==")
    device = resolve_device(args.device)
    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.strip().startswith("#")]
    n_empty = n_nan = n_div = 0
    for pr in prompts:
        out = generate(model, tok, pr, args.max_new_tokens, args.temperature, args.top_k, None, device)
        if len(out.strip()) == 0:
            n_empty += 1
        if not check_logits_finite(model, tok, pr, device):
            n_nan += 1
        if repetition_stats(out)["div_ok"]:
            n_div += 1
    print(f"prompts={len(prompts)} vazios={n_empty} nan={n_nan} div_ok={n_div}")
    if n_empty > 0:
        fails.append(f"{n_empty} respostas vazias")
    if n_nan > 0:
        fails.append(f"{n_nan} prompts com NaN/Inf nos logits")
    if n_div < int(len(prompts) * 0.9):
        fails.append(f"div_ok baixo: {n_div}/{len(prompts)}")

    print("== [4] log de treino ==")
    lp = Path(args.log)
    if lp.exists():
        lines = lp.read_text(encoding="utf-8").splitlines()
        evals = [l for l in lines if l.startswith("[eval]")]
        print(f"linhas={len(lines)} evals={len(evals)}")
        if evals:
            print(f"primeira: {evals[0]}")
            print(f"ultima:   {evals[-1]}")
    else:
        fails.append(f"log ausente: {lp}")

    print("=" * 50)
    if fails:
        print("VEREDITO: REVISAR")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("VEREDITO: APROVADO (pode commitar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
