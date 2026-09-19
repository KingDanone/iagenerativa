#!/usr/bin/env python3
"""Treinamento: python train.py [--config configs/gpu.json] [--device auto|cuda|cpu] [--resume ckpt.pt] [--dry-run]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import LMDataset, load_token_arrays  # noqa: E402
from src.model import build_model  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import train_loop  # noqa: E402
from src.utils import (  # noqa: E402
    count_parameters, format_params, gpu_info, load_checkpoint, resolve_device, set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/debug.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true", help="forward+VRAM check, sem treinar")
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device if args.device != "auto" else cfg.get("device", "auto"))
    cfg["_device_resolved"] = str(device)
    info = gpu_info()
    print(f"device={device} | {info}")

    # sincroniza vocab do modelo com o tokenizer real (evita mismatch embedding)
    try:
        _tok = BPETokenizer.load(cfg.get("tokenizer_path", "artifacts/tokenizer/tokenizer.json"))
        if len(_tok) != int(cfg.get("vocab_size", len(_tok))):
            print(f"ajuste vocab: config={cfg['vocab_size']} -> tokenizer={len(_tok)}")
            cfg["vocab_size"] = len(_tok)
    except FileNotFoundError:
        print("AVISO: tokenizer nao encontrado; usando vocab da config")

    model = build_model(cfg)
    if cfg.get("compile"):
        try:
            model = torch.compile(model)
            print("torch.compile ativo")
        except Exception as e:
            print(f"torch.compile indisponivel ({e}); seguindo sem compilar")
    total, trainable = count_parameters(model)
    print(f"parametros: total={format_params(total)} treinaveis={format_params(trainable)} "
          f"pesos_fp32~{total*4/1024**2:.1f}MB | L={cfg.get('n_layer')} H={cfg.get('n_head')} "
          f"d={cfg.get('n_embd')} ctx={cfg.get('block_size')} vocab={cfg.get('vocab_size')}")

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        cfg["_start_step"] = ckpt.get("step", 0)
        cfg["_opt_state"] = ckpt.get("optimizer_state")
        cfg["_best_val"] = ckpt.get("best_val", float("inf"))
        print(f"resume: {args.resume} step={cfg['_start_step']}")

    # dry run: forward curto p/ estimar VRAM antes de horas de treino (#111/#112)
    try:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            x = torch.randint(0, cfg["vocab_size"], (2, min(32, cfg["block_size"])), device=device)
            logits, loss = model(x, x)
        print(f"dry-run forward ok: logits={tuple(logits.shape)} loss={loss.item():.4f}")
        if device.type == "cuda":
            print(f"vram apos forward: alocado={torch.cuda.memory_allocated()/1024**3:.2f}GB "
                  f"reservado={torch.cuda.memory_reserved()/1024**3:.2f}GB")
        if args.dry_run:
            return
        model.train()
    except RuntimeError as e:
        print(f"FALHA no dry-run (provavel OOM). Veja docs/TRAINING.md #80. Erro: {e}")
        raise

    train_toks, val_toks = load_token_arrays(cfg.get("data_dir", "data/processed"))
    print(f"tokens: train={len(train_toks):,} val={len(val_toks):,}")
    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(LMDataset(train_toks, cfg["block_size"]), batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True)
    val_loader = DataLoader(LMDataset(val_toks, cfg["block_size"]), batch_size=bs, shuffle=False,
                            num_workers=nw, drop_last=True)
    eff = bs * int(cfg.get("gradient_accumulation_steps", 1))
    print(f"batch fisico={bs} acumulacao={cfg.get('gradient_accumulation_steps',1)} batch efetivo={eff}")
    stats = train_loop(cfg, model, train_loader, val_loader)
    print(f"treino concluido: {stats}")


if __name__ == "__main__":
    main()
