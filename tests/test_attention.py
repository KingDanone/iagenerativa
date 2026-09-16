import torch

from src.attention import CausalSelfAttention, causal_mask


def test_causal_mask_shape_and_values():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    # linha 1 (2a posicao) acessa cols 0,1 mas nao 2,3
    assert m[1, 0] and m[1, 1] and not m[1, 2] and not m[1, 3]
    assert m[0, 0] and not m[0, 1]


def test_attention_no_future_leak():
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=8, n_head=2).eval()
    B, T, C = 1, 4, 8
    x = torch.randn(B, T, C)
    with torch.no_grad():
        y1 = attn(x)
    # perturba apenas o futuro (pos 3); saidas das pos 0..2 nao podem mudar
    x2 = x.clone()
    x2[:, 3, :] += 100.0
    with torch.no_grad():
        y2 = attn(x2)
    assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-5), "vazamento de futuro!"


def test_multinead_divisibility_error():
    import pytest
    with pytest.raises(AssertionError):
        CausalSelfAttention(n_embd=7, n_head=2)
