import torch

from src.model import MiniGPT
from src.utils import load_checkpoint, save_checkpoint


def test_checkpoint_save_resume(tmp_path):
    m = MiniGPT(vocab_size=32, block_size=8, n_embd=16, n_head=2, n_layer=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    _, loss = m(x, x)
    loss.backward()
    opt.step()
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, {"model_state": m.state_dict(), "optimizer_state": opt.state_dict(),
                        "step": 7, "config": {"vocab_size": 32}, "train_loss": 1.0, "val_loss": 1.0})
    ckpt = load_checkpoint(p, map_location="cpu")
    assert ckpt["step"] == 7
    m2 = MiniGPT(vocab_size=32, block_size=8, n_embd=16, n_head=2, n_layer=1)
    m2.load_state_dict(ckpt["model_state"])
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a, b)
