"""Transformer decoder-only (MiniGPT), implementacao propria e explicavel.

  Token IDs -> Token Embedding + Positional Embedding
    -> Nx TransformerBlock (Pre-LN + CausalAttention + Residual + MLP + Residual)
    -> Final LayerNorm -> LM Head -> Logits [B,T,V]

Pre-LN: x = x + attn(norm1(x)); x = x + mlp(norm2(x)).
Estavel e facil de explicar (ver docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(n_embd * mlp_ratio)
        self.fc1 = nn.Linear(n_embd, hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        block_size: int = 256,
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) % n_head ({n_head}) != 0"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)  # posicional aprendivel
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, mlp_ratio, dropout) for _ in range(n_layer)]
        )
        self.norm_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.grad_checkpoint = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def enable_gradient_checkpointing(self) -> None:
        """Recomputa ativacoes no backward: -~60% VRAM por +~20% tempo."""
        self.grad_checkpoint = True

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequencia T={T} > block_size={self.block_size}"
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        if self.grad_checkpoint and self.training:
            import torch.utils.checkpoint as ckpt
            for blk in self.blocks:
                x = ckpt.checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def approx_size_mb(self) -> float:
        return self.count_parameters() * 4 / 1024**2  # fp32


def build_model(cfg: dict) -> MiniGPT:
    m = MiniGPT(
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg.get("n_embd", 256)),
        n_head=int(cfg.get("n_head", 4)),
        n_layer=int(cfg.get("n_layer", 6)),
        mlp_ratio=int(cfg.get("mlp_ratio", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    if cfg.get("grad_checkpoint"):
        m.enable_gradient_checkpointing()
    return m
