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
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--max-context-tokens", type=int, default=512)
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
    sess = ChatSession(max_context_tokens=args.max_context_tokens, encode_fn=tok.encode)

    print("========================================")
    print("        MINI IA GENERATIVA")
    print("========================================")
    print(f"\nModelo: MiniGPT ({format_params(total)}) | Device: {device}")
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
            print(f"modelo=MiniGPT params={format_params(total)} device={device} "
                  f"temp={args.temperature} top_k={args.top_k} "
                  f"ctx_tokens={sess.context_tokens(user)} max_ctx={args.max_context_tokens}")
            continue
        if user.startswith("/"):
            print("comando desconhecido. /help")
            continue
        sess.truncate()
        prompt = sess.build_prompt(user)
        # garante janela: trunca prompt a esquerda se preciso
        ids = tok.encode(prompt)
        if len(ids) > args.max_context_tokens:
            ids = ids[-args.max_context_tokens:]
            prompt = tok.decode(ids)
        with torch.no_grad():
            resp = generate(model, tok, prompt, args.max_new_tokens, args.temperature,
                            args.top_k, None, device).strip()
        sess.add_turn(user, resp)
        sess.truncate()
        print(f"\nIA: {resp}\n")


if __name__ == "__main__":
    main()
