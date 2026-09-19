# Fase 3 v2 — Base maior + foco conversacional (100% do zero)

Nada aqui usa peso pré-treinado, biblioteca de modelo ou API externa.
Só **dados públicos** alimentam o tokenizer, o Transformer e o loop de treino próprios.

## Pipeline v2 (rodando)

```bash
# 1) download 2GB: fineweb2-pt 50% + wiki 15% + carolina 10% + oscar 5%
#    + QA 8% + aya-pt 7% + oasst-pt 4% + seed
.venv/bin/python scripts/download_data.py --max-gb 2.0 --output data/raw/corpus_v2.jsonl
# 2) limpeza (unicode/NFC, dedup sha256, filtro PT, blocklist spam)
.venv/bin/python scripts/clean_data.py --input data/raw/corpus_v2.jsonl --output data/raw/corpus_v2.clean.jsonl
# 3) manifesto com licenças por fonte
.venv/bin/python scripts/prepare_data.py --raw data/raw/corpus_v2.jsonl \
  --clean data/raw/corpus_v2.clean.jsonl --out metadata/dataset_manifest_v2.json
# 4) tokenizer v2 (BPE próprio, vocab 8192, ~60M chars)
.venv/bin/python scripts/train_tokenizer.py --input data/raw/corpus_v2.clean.jsonl \
  --output artifacts/tokenizer/tokenizer_v2.json --vocab-size 8192
# 5) dataset (docs embaralhados antes do split; val 5% não-viesada)
.venv/bin/python scripts/build_dataset.py --input data/raw/corpus_v2.clean.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer_v2.json --out-dir data/processed_v2
```

## Treino base v2 (`configs/gpu_v2.json`)

~90M params (d640, L16, H10, ctx512, vocab 8k), 16000 steps × 64 × 512
= **~524M tokens (~17-20h na RTX 3050 6GB)**, bf16 + torch.compile +
grad-checkpointing + AdamW fused. VRAM medida: ~1,9/5,7 GB.

```bash
# inicia (ou retoma com --resume artifacts/checkpoints_v2/latest.pt)
.venv/bin/python train.py --config configs/gpu_v2.json --device cuda
# monitora
tail -f logs_v2/train.log
```

Checkpoints (`artifacts/checkpoints_v2/`): `latest.pt` sempre, `best.pt` na
melhor val. Early stopping: patience 8 evals. Baseline Fase 2 para comparar:
`artifacts/evaluation/eval_2026-09-18T22-34-08.json` (20/20 ok, 0 NaN).

## Depois: SFT conversacional (Fase E, ainda não implementada)

Continued training do `best.pt` v2 com mix ~80% conversa / 20% replay,
LR ~10× menor e loss só nos tokens do assistente (labels `-100` nos turnos
do usuário) — implementação própria pendente em `src/training.py`.

## Licenças das fontes (atribuição)

- FineWeb-2 por_Latn: **ODC-By 1.0** (atribuição obrigatória)
- Wikipedia PT: CC BY-SA · aya_dataset (PT): Apache-2.0 · oasst1 (PT): Apache-2.0
- Carolina/OSCAR/QA-PT: mistas — ver `metadata/dataset_manifest_v2.json`
