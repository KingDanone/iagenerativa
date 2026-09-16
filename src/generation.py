"""Amostragem e geracao autoregressiva."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next(logits_1d: torch.Tensor, temperature: float = 0.8, top_k: int | None = 40, top_p: float | None = None) -> int:
    """Amostra 1 token a partir de logits [V]. temperature<=0 => argmax."""
    assert logits_1d.ndim == 1
    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits_1d).item())
    logits = logits_1d.float() / max(float(temperature), 1e-6)
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        vals, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < vals[-1], torch.tensor(float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        order = torch.argsort(logits, descending=True)
        sorted_logits = logits[order]
        probs_s = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs_s, dim=-1)
        cutoff = torch.searchsorted(cum, torch.tensor(top_p)).item()
        keep = order[: int(cutoff) + 1]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[keep] = False
        logits = logits.masked_fill(mask, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    # fallback numerico
    if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
        return int(torch.argmax(logits_1d).item())
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    device: str | torch.device = "cpu",
    add_bos: bool = False,
    stop_at_eos: bool = True,
    verbose: bool = False,
) -> str:
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    ids = tokenizer.encode(prompt, add_bos=add_bos)
    # trunca prompt a esquerda p/ caber no contexto
    ctx = getattr(model, "block_size", 256)
    ids = ids[-(ctx - 1):] if len(ids) >= ctx else ids
    out = list(ids)
    eos = getattr(tokenizer, "eos_id", None)
    for _ in range(max_new_tokens):
        window = out[-(ctx):]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = sample_next(logits[0, -1], temperature, top_k, top_p)
        out.append(nxt)
        if verbose:
            print(tokenizer.decode([nxt]), end="", flush=True)
        if stop_at_eos and eos is not None and nxt == eos:
            break
    gen_ids = out[len(ids):]
    # remove EOS final p/ exibicao
    if gen_ids and eos is not None and gen_ids[-1] == eos:
        gen_ids = gen_ids[:-1]
    return tokenizer.decode(gen_ids)


def repetition_stats(text: str) -> dict:
    """Diagnostico simples de repeticao (ver especificacao #100)."""
    import re
    from collections import Counter

    words = re.findall(r"\S+", text.lower())
    wc = Counter(words)
    toks = text.split()
    rep_word = max(wc.values()) / max(len(words), 1) if words else 0.0
    # maior run do mesmo token adjacente
    max_run, run = 1, 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        max_run = max(max_run, run)
    return {
        "n_words": len(words),
        "n_unique": len(wc),
        "top_repeat_ratio": round(rep_word, 3),
        "max_adjacent_run": max_run,
        "div_ok": rep_word < 0.3 and max_run < 6,
    }
