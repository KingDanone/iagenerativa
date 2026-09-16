import pytest
import torch

from src.model import MiniGPT


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = MiniGPT(vocab_size=64, block_size=16, n_embd=32, n_head=4, n_layer=2)
    x = torch.randint(0, 64, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)


def test_overfit_tiny():
    """Modelo deve reduzir fortemente a loss num dataset minusculo (#43)."""
    torch.manual_seed(42)
    m = MiniGPT(vocab_size=32, block_size=8, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (4, 8))
    _, l0 = m(x, x)
    l0v = l0.item()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        _, loss = m(x, x)
        loss.backward()
        opt.step()
    _, l1 = m(x, x)
    assert l1.item() < l0v * 0.6, f"nao overfitou: {l0v:.3f} -> {l1.item():.3f}"


def test_gpu_forward():
    if not torch.cuda.is_available():
        pytest.skip("sem CUDA")
    m = MiniGPT(vocab_size=64, block_size=16, n_embd=32, n_head=4, n_layer=2).cuda()
    x = torch.randint(0, 64, (2, 16)).cuda()
    logits, loss = m(x, x)
    assert logits.is_cuda and torch.isfinite(loss)
    print(f"\n[VRAM teste] alocado={torch.cuda.memory_allocated()/1024**2:.1f}MB "
          f"reservado={torch.cuda.memory_reserved()/1024**2:.1f}MB")
