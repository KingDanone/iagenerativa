"""Tokenizer BPE proprio, controlado pelo projeto.

Objetivo: vocab 8k, excelente cobertura de portugues (acentos, cedilha).
Implementacao explicavel (nao usa `tokenizers`/`sentencepiece` no treino).

Formato:
  - nivel de palavra: o texto e dividido em blocos `\\S+|\\s+` (palavras vs
    blocos de espacos). Blocos de espaco (" ", "\\n\\n", ...) sao palavras
    como outras: merges aprendem "  ", "\\n\\n" etc. como 1 token.
  - merges BPE sao aprendidos *dentro* de palavras; treino usa indice
    invertido par->palavras + heap (O(merges afetados), nao O(corpus) por iter).
  - base = especiais + todos os caracteres unicos do corpus de treino.
  - merges iterativos pelo par adjacente mais frequente (ponderado por freq da palavra).

API: encode / decode / save / load / vocab_size.
Especiais: <PAD> <UNK> <BOS> <EOS>
"""
from __future__ import annotations

import heapq
import itertools
import json
import re
from collections import Counter
from pathlib import Path

SPLIT_RE = re.compile(r"\S+|\s+")

DEFAULT_SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

TOKENIZER_VERSION = 2
_ENCODE_CACHE_MAX = 300_000


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
        self._encode_cache: dict[str, list[str]] = {}

    # ---------- treino ----------
    def train(self, texts, vocab_size: int | None = None, min_freq: int = 2, max_words: int = 300_000) -> "BPETokenizer":
        """Aprende o vocabulario BPE a partir de um iteravel de strings."""
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        self._encode_cache = {}
        word_freq: Counter[str] = Counter()
        for t in texts:
            if not t:
                continue
            for chunk in SPLIT_RE.findall(t):
                # palavra OU bloco de espacos inteiro (" ", "\n\n", ...):
                # blocos frequentes viram 1 token via merges
                word_freq[chunk] += 1
                if len(word_freq) >= max_words * 2:
                    break
            if len(word_freq) >= max_words * 2:
                break
        # limita as palavras mais frequentes p/ treino rapido e estavel
        if len(word_freq) > max_words:
            word_freq = Counter(dict(word_freq.most_common(max_words)))

        # filtra palavras rarissimas (mantem atomos de 1 char sempre)
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
        weight: dict[str, int] = dict(filt)

        def word_pair_counts(syms: list[str]) -> Counter[tuple[str, str]]:
            return Counter(zip(syms, syms[1:]))

        # indice invertido: par -> {palavras que o contem}; freq ponderada
        pair_freq: dict[tuple[str, str], int] = {}
        where: dict[tuple[str, str], set[str]] = {}
        for w, syms in splits.items():
            c = weight[w]
            for p, k in word_pair_counts(syms).items():
                pair_freq[p] = pair_freq.get(p, 0) + k * c
                s = where.get(p)
                if s is None:
                    where[p] = {w}
                else:
                    s.add(w)
        # heap (-freq, ordem) com validacao preguicosa: so os pares afetados
        # por um merge sao reprocessados (antes: scan O(corpus) por iteracao)
        tick = itertools.count()
        heap: list[tuple[int, int, tuple[str, str]]] = [(-f, next(tick), p) for p, f in pair_freq.items()]
        heapq.heapify(heap)

        merges: list[list[str]] = []

        def forget_word_pairs(w: str, syms: list[str]) -> None:
            c = weight[w]
            for p, k in word_pair_counts(syms).items():
                f = pair_freq.get(p)
                if f is None:
                    continue
                f -= k * c
                if f <= 0:
                    pair_freq.pop(p, None)
                else:
                    pair_freq[p] = f
                s = where.get(p)
                if s is not None:
                    s.discard(w)
                    if not s:
                        where.pop(p, None)

        def remember_word_pairs(w: str, syms: list[str]) -> None:
            c = weight[w]
            for p, k in word_pair_counts(syms).items():
                pair_freq[p] = pair_freq.get(p, 0) + k * c
                s = where.get(p)
                if s is None:
                    where[p] = {w}
                else:
                    s.add(w)
                heapq.heappush(heap, (-pair_freq[p], next(tick), p))

        while len(vocab) < self.vocab_size and heap:
            neg, _, (a, b) = heapq.heappop(heap)
            f = pair_freq.get((a, b))
            if f is None or -neg != f:
                continue  # entrada obsoleta
            if f < min_freq and len(vocab) >= self.vocab_size // 2:
                break
            new_tok = a + b
            if new_tok not in vocab:
                vocab[new_tok] = len(vocab)
            merges.append([a, b])
            if len(merges) % 500 == 0 or len(vocab) >= self.vocab_size:
                print(f"  BPE {len(vocab)}/{self.vocab_size} merges={len(merges)} ultimo={new_tok!r} freq={f}", flush=True)
            # aplica merge so nas palavras que contem o par
            for w in list(where.get((a, b), ())):
                syms = splits[w]
                forget_word_pairs(w, syms)
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
                remember_word_pairs(w, out)

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
        hit = self._encode_cache.get(word)
        if hit is not None:
            return hit
        if word in self.token_to_id:
            out = [word]
        else:
            syms = list(word)
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
            out = syms
        if len(self._encode_cache) >= _ENCODE_CACHE_MAX:
            self._encode_cache.clear()  # teto de RAM; Zipf reaquece rapido
        self._encode_cache[word] = out
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if not text:
            if add_eos:
                ids.append(self.eos_id)
            return ids
        for chunk in SPLIT_RE.findall(text):
            # palavra ou bloco de espacos: mesma segmentacao do treino
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
            "version": TOKENIZER_VERSION,
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
        tok._encode_cache = {}
        tok._sync_special_ids()
        return tok

    def __len__(self) -> int:
        return len(self.token_to_id)
