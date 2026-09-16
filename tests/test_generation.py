import torch

from src.generation import repetition_stats, sample_next
from src.model import MiniGPT
from src.tokenizer import BPETokenizer


def _tiny_setup():
    tok = BPETokenizer(vocab_size=64).train(
        ["Olá mundo. " * 5, "Python é legal. " * 5, "O Brasil é grande. " * 5] * 10, vocab_size=64)
    model = MiniGPT(vocab_size=len(tok), block_size=16, n_embd=32, n_head=4, n_layer=1)
    return tok, model


def test_sample_next_greedy_and_sampling():
    logits = torch.tensor([1.0, 5.0, 2.0])
    assert sample_next(logits, temperature=0.0) == 1
    outs = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(30)}
    assert outs <= {1, 2}


def test_generate_returns_text_and_stops():
    import torch
    from src.generation import generate
    torch.manual_seed(0)
    tok, model = _tiny_setup()
    out = generate(model, tok, "Olá", max_new_tokens=10, temperature=0.0)
    assert isinstance(out, str)


def test_repetition_stats():
    ok = repetition_stats("O Brasil é um país grande e diverso")
    assert ok["div_ok"]
    bad = repetition_stats("azul azul azul azul azul azul azul azul azul azul")
    assert not bad["div_ok"]
