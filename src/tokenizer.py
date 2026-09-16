"""Tokenizer BPE proprio, controlado pelo projeto.

Objetivo: vocab ~4096, excelente cobertura de portugues (acentos, cedilha).
Implementacao explicavel (nao usa `tokenizers`/`sentencepiece` no treino).

Formato:
  - nivel de palavra: o texto e dividido em blocos `\\S+|\\s+` (palavras vs espacos).
  - merges BPE sao aprendidos *dentro* de palavras; espacos/quebras sao atomos.
  - base = especiais + todos os caracteres unicos do corpus de treino.
  - merges iterativos pelo par adjacente mais frequente (ponderado por freq da palavra).

API: encode / decode / save / load / vocab_size.
Especiais: <PAD> <UNK> <BOS> <EOS>
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPLIT_RE = re.compile(r"\S+|\s+")

DEFAULT_SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class BPETokenizer:
    def __init__(
        self,
        vocab_size: int = 4096,
        specials: list[str] | None = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.specials = list(specials or DEFAULT_SPECIALS)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[list[str]] = []  # lista de [a, b] na ordem aprendida
        self.merge_rank: dict[tuple[str, str], int] = {}
        # ids especiais (preenchidos em _rebuild)
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    # ---------- treino ----------
    def train(self, texts, vocab_size: int | None = None, min_freq: int = 2, max_words: int = 300_000) -> "BPETokenizer":
        """Aprende o vocabulario BPE a partir de um iteravel de strings."""
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        word_freq: Counter[str] = Counter()
        n_docs = 0
        for t in texts:
            if not t:
                continue
            n_docs += 1
            for chunk in SPLIT_RE.findall(t):
                if chunk.strip() == "":
                    # cada caractere de espaco conta como "palavra" de 1 char
                    for ch in chunk:
                        word_freq[ch] += 1
                else:
                    word_freq[chunk] += 1
                if len(word_freq) >= max_words * 2:
                    break
            if n_docs % 20000 == 0 and len(word_freq) > max_words:
                break
        # limita as palavras mais frequentes p/ treino rapido e estavel
        if len(word_freq) > max_words:
            word_freq = Counter(dict(word_freq.most_common(max_words)))

        # filtra palavras rarissimas (mantem chars de espaco sempre)
        filt = Counter({w: c for w, c in word_freq.items() if c >= min_freq or len(w) == 1})
        if not filt:
            filt = word_freq

        # vocab base: especiais + caracteres unicos
        chars: set[str] = set()
        for w in filt:
            chars.update(list(w))
        vocab: dict[str, int] = {}
        for s in self.specials:
            if s not in vocab:
                vocab[s] = len(vocab)
        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # representacao mutavel das palavras: lista de simbolos
        splits: dict[str, list[str]] = {w: list(w) for w in filt}
        merges: list[list[str]] = []

        n_target = self.vocab_size - len(vocab)
        done = 0
        while len(vocab) < self.vocab_size:
            pair_freq: Counter[tuple[str, str]] = Counter()
            for w, c in filt.items():
                syms = splits[w]
                for i in range(len(syms) - 1):
                    pair_freq[(syms[i], syms[i + 1])] += c
            if not pair_freq:
                break
            (a, b), freq = pair_freq.most_common(1)[0]
            if freq < min_freq:
                # ainda permite merges se vocab muito pequeno? nao: evita lixo
                # mas se faltar muito vocab, relaxa uma vez
                if len(vocab) < self.vocab_size // 2:
                    pass
                else:
                    break
            new_tok = a + b
            if new_tok in vocab:
                # par ja fundido conceitualmente; remove ocorrencias p/ avancar
                # (evita loop infinito)
                for w in list(splits.keys()):
                    syms = splits[w]
                    out = []
                    i = 0
                    while i < len(syms):
                        if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                            out.append(new_tok)
                            i += 2
                        else:
                            out.append(syms[i])
                            i += 1
                    splits[w] = out
                continue
            vocab[new_tok] = len(vocab)
            merges.append([a, b])
            done += 1
            if done % 500 == 0 or len(vocab) >= self.vocab_size:
                print(f"  BPE {len(vocab)}/{self.vocab_size} merges={len(merges)} ultimo={new_tok!r} freq={freq}", flush=True)
            # aplica merge em todas as palavras
            for w in list(splits.keys()):
                syms = splits[w]
                if a not in syms:
                    continue
                out: list[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                splits[w] = out

        self.token_to_id = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.merge_rank = {(a, b): r for r, (a, b) in enumerate(merges)}
        self._sync_special_ids()
        return self

    def _sync_special_ids(self) -> None:
        g = self.token_to_id.get
        if "<PAD>" in self.token_to_id:
            self.pad_id = g("<PAD>")  # type: ignore
        if "<UNK>" in self.token_to_id:
            self.unk_id = g("<UNK>")  # type: ignore
        if "<BOS>" in self.token_to_id:
            self.bos_id = g("<BOS>")  # type: ignore
        if "<EOS>" in self.token_to_id:
            self.eos_id = g("<EOS>")  # type: ignore

    # ---------- encode / decode ----------
    def _encode_word(self, word: str) -> list[str]:
        if word in self.token_to_id:
            return [word]
        syms = list(word)
        if len(syms) == 1:
            return syms
        # aplica merges em ordem de rank (guloso): sempre o par de menor rank
        while len(syms) >= 2:
            best = None
            best_rank = None
            for i in range(len(syms) - 1):
                r = self.merge_rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = i
            if best is None:
                break
            i = best
            syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2 :]
        return syms

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if not text:
            if add_eos:
                ids.append(self.eos_id)
            return ids
        for chunk in SPLIT_RE.findall(text):
            if chunk.strip() == "":
                for ch in chunk:
                    ids.append(self.token_to_id.get(ch, self.unk_id))
            else:
                for piece in self._encode_word(chunk):
                    ids.append(self.token_to_id.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        skip = {self.token_to_id.get("<PAD>"), self.token_to_id.get("<BOS>"), self.token_to_id.get("<EOS>")}
        for i in ids:
            tok = self.id_to_token.get(int(i), "<UNK>")
            if int(i) in skip or tok in ("<PAD>", "<BOS>", "<EOS>"):
                continue
            if tok == "<UNK>":
                parts.append("�")
            else:
                parts.append(tok)
        return "".join(parts)

    # ---------- persistencia ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "specials": self.specials,
            "vocab": self.token_to_id,
            "merges": self.merges,
            "version": 1,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=payload.get("vocab_size", len(payload["vocab"])),
                  specials=payload.get("specials", DEFAULT_SPECIALS))
        tok.token_to_id = dict(payload["vocab"])
        tok.id_to_token = {int(i): t for t, i in tok.token_to_id.items()}
        tok.merges = [list(m) for m in payload.get("merges", [])]
        tok.merge_rank = {(a, b): r for r, (a, b) in enumerate(tok.merges)}
        tok._sync_special_ids()
        return tok

    def __len__(self) -> int:
        return len(self.token_to_id)
