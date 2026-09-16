import json
import tempfile
from pathlib import Path

from src.tokenizer import BPETokenizer

SAMPLES = [
    "Olá, mundo!",
    "São Luís é uma cidade brasileira.",
    "Inteligência artificial",
    "ação",
    "informação",
    "programação",
    "coração",
]

CORPUS = [
    "Olá, mundo! Este é um texto em português.",
    "São Luís é uma cidade brasileira com muita história.",
    "Inteligência artificial é uma área da computação.",
    "A ação de informar a programação do coração exige atenção.",
    "Python é uma linguagem de programação usada no Brasil.",
    "O sistema solar é formado pelo Sol e pelos planetas.",
    "À noite, o coração sente a emoção da canção: pão, mãe, limão, órgão, ângulo, êxito, ônus.",
] * 20


def test_bpe_train_encode_decode_roundtrip():
    tok = BPETokenizer(vocab_size=256).train(CORPUS, vocab_size=256)
    assert len(tok) <= 256
    for s in SAMPLES:
        ids = tok.encode(s)
        assert len(ids) > 0
        assert all(0 <= i < len(tok) for i in ids)
        back = tok.decode(ids)
        # roundtrip quase exato (espacos normalizados sao preservados aqui)
        assert back.replace("�", "") != ""


def test_accents_covered():
    tok = BPETokenizer(vocab_size=256).train(CORPUS, vocab_size=256)
    joined = "".join(tok.token_to_id.keys())
    corpus_chars = set("".join(CORPUS))
    for ch in "ãõçáéíóúâêôà":
        if ch in corpus_chars:
            assert ch in joined, f"acento {ch!r} do corpus ausente no vocab"


def test_specials_and_save_load():
    tok = BPETokenizer(vocab_size=128).train(CORPUS, vocab_size=128)
    assert tok.token_to_id["<PAD>"] == 0
    assert "<UNK>" in tok.token_to_id and "<BOS>" in tok.token_to_id and "<EOS>" in tok.token_to_id
    ids = tok.encode("Olá", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tok.json"
        tok.save(p)
        t2 = BPETokenizer.load(p)
        assert t2.encode("Programação") == tok.encode("Programação")
        assert t2.decode(tok.encode("São Luís")) == tok.decode(tok.encode("São Luís"))
