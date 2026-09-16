# Matemática — mini_ia_generativa

Explicação acessível dos fundamentos. Implementação em `src/attention.py`, `src/model.py`, `src/training.py`.

## 1. Embedding

Cada token (inteiro `t`) vira um vetor denso:

```
x = E[t]          # E: [vocab, d], aprendida
```

Somamos o embedding posicional (posição `p`, também aprendido):

```
h0 = E[t] + P[p]
```

Sem isso, a atenção não saberia a ordem das palavras.

## 2. Attention (auto-atenção causal)

Para a sequência `X` (`[T, d]`), três projeções lineares:

```
Q = X Wq     K = X Wk     V = X Wv
```

Pesos de atenção (uma head, dimensão `d_k`):

```
S = Q K^T / sqrt(d_k)        # [T, T] escores
A = softmax(S + máscara)     # máscara = -inf no triângulo superior
saída = A V
```

A divisão por `sqrt(d_k)` evita que o softmax sature quando `d_k` cresce.
A máscara causal garante: posição `i` só usa `j ≤ i` (teste em `tests/test_attention.py`
perturba o futuro e confirma que o passado não muda).

Multi-head: divide `d` em `h` heads (`d % h == 0`), concatena e projeta com `Wo`.
Cada head aprende um tipo de relação (sintaxe, concordância, tópico...).

## 3. Language modeling

O modelo estima, para cada posição `t`:

```
P(x_t | x_1, ..., x_{t-1})
```

Na prática: logits `z` (`[V]`) → `softmax(z)` = distribuição sobre o vocabulário.
Geração = amostrar dessa distribuição, anexar o token e repetir.

## 4. Loss (cross-entropy)

Para o token correto `y` com probabilidade prevista `p_y`:

```
loss = -log p_y = -z_y + log Σ_j exp(z_j)
```

Média sobre batch × posições. Minimizar a loss = maximizar a probabilidade
atribuída aos tokens reais do corpus.

## 5. Perplexidade

```
perplexidade = exp(loss)
```

Interpretação: "em média, o modelo hesita entre ~PPL tokens".
PPL 80 → escolhe entre ~80 plausíveis; PPL 400+ → quase uniforme (início do treino).

## 6. Backpropagation + AdamW (conceitual)

1. Forward calcula loss.
2. Backprop aplica a regra da cadeia e obtém `∂loss/∂w` para cada peso.
3. AdamW atualiza cada peso com passo adaptativo (momento + variância) e
   decaimento de peso desacoplado (regularização).
4. O schedule aquece a taxa (warmup linear) e depois decai por cosseno até `min_lr`.

## 7. Residual + Pre-LN

```
x = x + atenção(norm1(x))
x = x + mlp(norm2(x))
```

O caminho residual carrega o gradiente sem atenuação por `N` camadas
(por isso redes de 8 blocos treinam); a LayerNorm antes do bloco (Pre-LN)
mantém as ativações estáveis desde o passo 1.
