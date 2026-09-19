#!/usr/bin/env python3
"""Chat CLI: python chat.py [--checkpoint ...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chat import HELP, ChatSession  # noqa: E402
from src.generation import generate  # noqa: E402
from src.model import build_model  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.utils import count_parameters, format_params, load_checkpoint, resolve_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--max-context-tokens", type=int, default=None,
                   help="limite de contexto do chat; default = block_size do checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    tok = BPETokenizer.load(args.tokenizer)
    total, _ = count_parameters(model)
    # contexto do chat nunca pode exceder o block_size do modelo (#ctx-mismatch)
    block = int(cfg.get("block_size", 256))
    ctx_limit = int(args.max_context_tokens) if args.max_context_tokens else block
    if ctx_limit > block:
        print(f"AVISO: max-context-tokens={ctx_limit} > block_size={block}; limitando a {block}")
        ctx_limit = block
    sess = ChatSession(max_context_tokens=ctx_limit, encode_fn=tok.encode)

    print("========================================")
    print("        GUARÁ — IA GENERATIVA")
    print("========================================")
    print(f"\nModelo: Guará ({format_params(total)}) | Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("\nDigite /help para ajuda.\nDigite /exit para sair.\n")
    while True:
        try:
            user = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break
        if not user:
            continue
        if user == "/exit":
            print("Encerrando.")
            break
        if user == "/help":
            print(HELP)
            continue
        if user == "/reset":
            sess.reset()
            print("(contexto limpo)")
            continue
        if user == "/history":
            print(sess.render_history())
            continue
        if user == "/stats":
            print(f"modelo=Guará params={format_params(total)} device={device} "
                  f"temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
                  f"ctx_tokens={sess.context_tokens(user)} max_ctx={ctx_limit} block={block}")
            continue
        if user.startswith("/"):
            print("comando desconhecido. /help")
            continue
        sess.truncate()
        prompt = sess.build_prompt(user)
        # garante janela: trunca prompt a esquerda se preciso
        ids = tok.encode(prompt)
        if len(ids) > ctx_limit:
            ids = ids[-ctx_limit:]
            prompt = tok.decode(ids)
        with torch.no_grad():
            resp = generate(model, tok, prompt, args.max_new_tokens, args.temperature,
                            args.top_k, args.top_p, device).strip()
        sess.add_turn(user, resp)
        sess.truncate()
        print(f"\nIA: {resp}\n")


if __name__ == "__main__":
    main()
