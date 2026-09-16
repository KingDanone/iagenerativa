"""Self-attention causal manual (sem nn.Transformer).

  Q = X Wq,  K = X Wk,  V = X Wv
  A = softmax(QK^T / sqrt(d_k) + mask) V

Mascara causal: posicao i so acessa j <= i (triangular inferior).
Ver docs/MATH.md e tests/test_attention.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device=None) -> torch.Tensor:
    """[T,T] bool: True onde a atencao e PERMITIDA (j <= i)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) deve ser divisivel por n_head ({n_head})"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.split(self.n_embd, dim=2)
        # [B,nh,T,hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,nh,T,T]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v  # [B,nh,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
