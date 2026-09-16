# Resultados (todos medidos — nada inventado)

Data do treino principal: 2026-09-13. Logs: `logs/train_full.log`.

## Hardware / software

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop (5,67 GB utilizáveis, cap. 8.6)
- CPU: Intel i5-12450HX (12 threads), RAM 15 GB
- Python 3.12.3, PyTorch 2.14.0+cu130, CUDA runtime 13.0, driver 595.84

## Dataset

| Métrica | Valor |
|---|---|
| Fontes | Wikipedia PT (5978 docs) + QA-PT `Jpzinn654/qa-portuguese-small` (19178) + seed próprio (12) |
| Docs (após limpeza) | 25.168 (raw 25.310; 28 descartados, 114 dups) |
| Chars | 70.868.855 |
| Tokens BPE-4096 | 31.756.233 (train 30.168.422 / val 1.587.811) |
| Manifesto | `metadata/dataset_manifest.json` |

## Tokenizer

BPE próprio, vocab 4096, merges 3822, especiais `<PAD> <UNK> <BOS> <EOS>`.
Roundtrip exato em "Olá, mundo!", "São Luís é uma cidade brasileira.",
"ação", "informação", "programação", "coração", "Inteligência artificial".

## Modelo

MiniGPT decoder-only: vocab 4096, ctx 256, d=384, 6 heads, 8 layers, MLP 1536.
**Parâmetros: 17.431.296** (treináveis: 17.431.296), pesos fp32 ~66,5 MB.

## Treinamento

| Métrica | Valor |
|---|---|
| Steps | 5000 |
| Batch físico / acumulação / efetivo | 8 / 8 / 64 (16.384 tok/step) |
| LR | 3e-4, warmup 200, cosseno → 3e-5; AdamW wd=0.1; AMP fp16+scaler |
| Tokens vistos | 81.920.000 (~2,7 épocas) |
| Tempo | 2557,7 s (~42,6 min) |
| Tokens/s (médio) | 32.029 |
| VRAM máx. | 0,81 / 5,67 GB |
| Train loss (1º → último) | 8,3134 → 2,3229 |
| Val loss (melhor, step 4750) | **2,5523** (final 2,5663) |
| Perplexidade (melhor) | **~12,8** (final ~13,0) |
| Checkpoints | `best.pt` (val mín), `latest.pt`, `demo.pt` (=best), 200 MB c/ otimizador |

Curva (val): 4,00 → 3,56 → 3,33 → 3,14 → 2,97 → 2,93 → 2,82 → 2,81 →
2,73 → 2,72 → 2,66 → 2,64 → 2,64 → 2,59 → 2,59 → 2,57 → 2,56 →
2,57 → 2,55 → 2,57. Sem overfitting explosivo (train 2,32 vs val 2,55).

## Geração (`temperature=0.8, top-k=40, max 80`)

`artifacts/evaluation/eval_2026-09-13T17-28-13.json` — 20/20 prompts com saída,
0 vazios, 0 NaNs, `div_ok=True` nos 20. Exemplos REAIS:

- P "Python é uma linguagem" → R "...de programação que permite a redução em
  sistemas operacionais de programação eletrônica..." (início correto, deriva depois)
- P "São Luís é uma cidade" → R "...constituída por uma área de 800 habitantes
  por km2..." (estilo enciclopédico, números alucinados)
- P "O que é Python?" → R "...é uma linguagem de texto que tem uma linguagem de
  programação por meio de dados..." (tópico certo, semântica fraca)
- P "Qual é o meu nome?" (após "Meu nome é João.") → resposta sem o nome
  (contexto além da capacidade atual — limitação documentada)

## Problemas encontrados (não escondidos)

1. **Spam no QA**: fonte `qa-portuguese-small` contém promo-spam ("códigos
   promocionais...", "catracalivre.com"); limpeza atual não filtra por domínio.
   Efeito visível nos prompts 9 e no chat. Ação futura: blocklist de domínios.
2. **Token `<UNK>` gerado**: trechos com alfabetos raros viram `�` (ex. prompt 5).
3. **Memória conversacional curta**: não retém nome após turnos (esperado p/ 17M/ctx256).
4. **Teardown crash** (`datasets`+`torch` no fim do `download_data.py`, "core dump"
   após o arquivo completo): inofensivo, arquivo íntegro; documentado.
5. Tokenizer lento no corpus cheio → amostragem estratificada 12M chars (mantém qualidade).

## Limitações

Modelo pequeno, 2,7 épocas (Chinchilla sugeriria ~10–20× mais tokens p/ 17M),
contexto 256, QA ruidoso, sem conhecimento atualizado, sem memória persistente.
Geração: português reconhecível em fragmentos, coerência global limitada —
compatível com a meta "básico, não-ChatGPT" da especificação §92.

## Experimentos

- `experiments/debug_fase1.json`: 628k params, 120 steps, loss 8,33→6,18.
- `experiments/gpu_fase2.json`: este treino. Comparação A<B confirma escala ajuda;
  próximo passo: 3–5× steps ou filtragem do QA antes de aumentar o modelo.
