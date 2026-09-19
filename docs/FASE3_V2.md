## O que foi treinado (realizado ✅)

Por decisão do autor (17M params / 26M tokens conversacionais), optou-se por um
ajuste fino conversacional **curto** em vez do treino base de 16k steps.

### Fase 3 — Ajuste Guará (3000 + 14000 steps, ~26 + ~90 min)

- **Dados**: `data/raw/corpus_chat.jsonl` (mix de 421k docs conversacionais PT +
  20% replay de conhecimento → 505k docs, 215M tokens). Script: `scripts/make_chat_mix.py`.
- **Início**: pesos da Fase 2 (`--init-from artifacts/checkpoints/best.pt`), LR 1e-4,
  schedule próprio.
- **Continuação**: 11000 steps extras com LR 5e-5 → total 14000 steps / 49M tokens.
- **Resultado final** (`artifacts/checkpoints_chat/best.pt`, step 13000):
  - best_val=0,689 → PPL=1,99 (val) | train_final=1,76
  - 20/20 prompts sem vazios, sem NaN, 20/20 `div_ok`
  - VRAM máx: 0,8/5,7 GB | throughput ~33k tok/s
- **Configuração**: `configs/chat_tune.json` (3000 steps) + `configs/chat_tune_cont.json` (continuação 14000).
- **Saída**: `artifacts/checkpoints_chat/best.pt`, `latest.pt`.

## Base maior (não realizada ⏸️)

O treino base de 16k steps (~21h, config `configs/gpu_v2.json`, ~90M params,
bf16 + compile + grad-checkpoint) foi lançado mas abortado pelo freeze de RAM
do notebook (ver abaixo). Ficou documentado em `configs/gpu_v2.json` para
execução futura. Se executar, o checkpoint de retoma é `artifacts/checkpoints_v2/latest.pt`
(caso exista).

## Freeze de RAM (incidente 19/09 ~00:19)

O notebook travou por **esgotamento de RAM** (não GPU):
- `journalctl -b -1` mostra `Under memory pressure` às 00:19:39 e `Power key pressed` às 00:19:42.
- Causa raiz: `RandomSampler(shuffle=True)` materializa `randperm(552M)` = 4,4GB +
  torch.compile (12 workers Triton) + baseline ~7GB do desktop.
- **Correções aplicadas**: sampler `replacement=True` (sem permutação),
  `load_token_arrays` mmap direto (1 shard), `num_workers=0`, `compile_threads=2`.
- **Lição**: treinos longos >1h devem usar sampler com replacement e limitar compile threads.

## Reprodutibilidade

```bash
# mix conversacional (não precisa re-downloadar, já está em data/raw/)
.venv/bin/python scripts/make_chat_mix.py
.venv/bin/python scripts/build_dataset.py --input data/raw/corpus_chat.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer.json --out-dir data/processed_chat
# ajuste fino (retoma do step 3000 com schedule novo)
.venv/bin/python train.py --config configs/chat_tune_cont.json --device cuda \
  --resume artifacts/checkpoints_chat/latest.pt
# ou do zero (Fase 3a = 3000 steps)
.venv/bin/python train.py --config configs/chat_tune.json --device cuda
# ou iniciais do zero (base 90M, 16k steps ~17h)
.venv/bin/python train.py --config configs/gpu_v2.json --device cuda
```