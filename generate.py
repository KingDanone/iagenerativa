#!/usr/bin/env python3
"""Geracao: python generate.py --prompt "O Brasil" [--checkpoint ...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.generation import generate  # noqa: E402
from src.model import build_model  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.utils import load_checkpoint, resolve_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--checkpoint", default="artifacts/checkpoints_chat/best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config") or (json.loads(Path(args.config).read_text()) if args.config else None)
    if cfg is None:
        raise SystemExit("checkpoint sem config e --config nao informado")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    tok = BPETokenizer.load(args.tokenizer)
    out = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature,
                   args.top_k, args.top_p, device)
    print(out)
    ver = ckpt.get("config", {}).get("version", "?")
    name = ckpt.get("config", {}).get("model_name", "Guará")
    print(f"[{name} v{ver}]")


if __name__ == "__main__":
    main()
