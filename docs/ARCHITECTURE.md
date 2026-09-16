# Arquitetura — MiniGPT (decoder-only)

```
Token IDs → Token Emb + Pos Emb → [TransformerBlock × N] → LayerNorm → LM Head
→ Logits [B, T, V] → próximo token
```

## Configuração real (`configs/gpu.json`)

| Hiperparâmetro | Valor | Papel |
|---|---|---|
| vocab_size | 4096 | BPE próprio PT (acentos preservados) |
| block_size | 256 | janela de contexto (configurável) |
| n_embd | 384 | largura dos vetores |
| n_head | 6 | heads de atenção (384/6=64 por head) |
| n_layer | 8 | blocos Transformer |
| mlp_ratio | 4 | MLP interna 384→1536→384 (GELU) |
| dropout | 0.1 | regularização |
| **parâmetros** | **17.431.296** | medido via `count_parameters` |
| pesos fp32 | ~66,5 MB | fp16 de treino ~33 MB |

Contagem aproximada: embeddings 4096×384 + 256×384 ≈ 1,67M;
por bloco ≈ 1,77M (atenção 4×384² + MLP 2×384×1536) ×8 ≈ 14,2M;
head 384×4096 ≈ 1,57M. Total ≈ 17,4M — dentro da meta 10–30M.

## Por que cada componente existe

- **Token/pos embedding**: transforma IDs em vetores ordenados (sem posição, "cão morde homem" = "homem morde cão").
- **CausalSelfAttention** (`src/attention.py`): mistura informação só do passado (máscara triangular); múltiplas heads capturam relações distintas.
- **MLP+GELU**: computação por posição; expande 4× para dar capacidade não-linear.
- **Pre-LN + residual**: estabilidade do gradiente em 8 camadas (ver `docs/MATH.md` §7).
- **LM Head sem viés**: projeta de volta ao vocabulário; saída `[B,T,V]`.

## O que NÃO usamos (1ª versão, de propósito)

RoPE, GQA, gradient checkpointing, torch.compile: a prioridade é código legível
para explicação acadêmica. VRAM medida (0,8 GB de 5,7 GB) mostra que não foram
necessários neste tamanho.

Arquivo principal: `src/model.py` (`CausalSelfAttention`, `MLP`, `TransformerBlock`, `MiniGPT`).
