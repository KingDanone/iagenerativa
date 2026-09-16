# Mini IA Generativa em Português

Este projeto implementa e treina do zero um pequeno modelo de linguagem
autoregressivo baseado em Transformer — **sem pesos pré-treinados, sem
fine-tuning, sem APIs externas**. Todo texto gerado sai do próprio modelo.

```
                  CORPUS
                    │
                    ▼
             Limpeza / Filtro
                    │
                    ▼
                 BPE (próprio, vocab 4096)
                    │
                    ▼
               Token IDs (uint16)
                    │
                    ▼
             Embeddings (token + posição)
                    │
                    ▼
        ┌─────────────────────┐
        │ Transformer Block ×8 │
        │                      │
        │ LayerNorm (Pre-LN)   │
        │ Causal Attention     │
        │ Residual             │
        │ LayerNorm            │
        │ MLP (GELU)           │
        │ Residual             │
        └─────────────────────┘
                    │
                    ▼
                LM Head
                    │
                    ▼
                  Logits
                    │
                    ▼
              Próximo Token
                    │
                    ▼
                 Texto
```

## 1. Visão geral

MiniGPT decoder-only de **17,4M parâmetros**, treinado em **31,7M tokens**
de português (Wikipedia + QA + seed dialógico), com chat CLI, avaliação
e documentação acadêmica completas.

## 2. Objetivo

Demonstrar de forma convincente o pipeline completo
dados → limpeza → BPE → dataset → Transformer → treino → geração → chat,
numa GPU de 6 GB. Não é um concorrente de LLMs comerciais.

## 3. Arquitetura

Ver `docs/ARCHITECTURE.md` e `docs/MATH.md`. Resumo:
`vocab 4096, contexto 256, d=384, 6 heads, 8 layers, MLP 1536, Pre-LN,
atenção causal manual, embeddings posicionais aprendíveis`.

## 4. Dataset

Ver `docs/DATASET_SELECTION.md` e `metadata/dataset_manifest.json`.
Fase 2: 25.168 docs / 70,8M chars: Wikipedia PT (~55% do orçamento),
QA português (~20%), seed dialógico próprio no formato do chat.
Limpeza: NFC, HTML, URLs, dedup sha256 normalizado, filtro PT.
Nada do corpus vai para o Git — só scripts + manifesto.

## 5. Tokenizer

BPE próprio (`src/tokenizer.py`, sem `tokenizers`/`sentencepiece` no treino),
vocab 4096, `encode/decode/save/load`, especiais `<PAD> <UNK> <BOS> <EOS>`.
Roundtrip: `"São Luís é uma cidade brasileira."` → idêntico.

## 6. Treinamento

Ver `docs/TRAINING.md`. AdamW 3e-4 + warmup/cosseno, batch efetivo 64
(8×8 acumulação), AMP fp16, 5000 steps (~82M tokens) na RTX 3050:
**~33k tok/s, VRAM ~0,8/5,7 GB**. Números finais em `docs/RESULTS.md`.

## 7. GPU

`--device auto|cuda|cpu` (padrão auto). `--dry-run` estima VRAM antes do
treino longo. Fallback CPU funciona para testes.

## 8. Geração

`python generate.py --prompt "O Brasil" --temperature 0.8 --top-k 40`
(para no `<EOS>`; `top-p` opcional).

## 9. Chat

`python chat.py` — mantém contexto curto (`/help /history /reset /stats /exit`).
Contexto ≠ memória persistente (ver `docs/CHAT.md`).

## 10. Avaliação

`eval/prompts_ptbr.txt` (20 prompts) + `python evaluate.py` →
`artifacts/evaluation/*.json` com saída, repetição e timestamp.

## 11. Instalação

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 12. Uso (reproduzível)

```bash
.venv/bin/python scripts/download_data.py --sources wikipedia,conversacional,seed --max-gb 0.12
.venv/bin/python scripts/clean_data.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train_tokenizer.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python train.py --config configs/gpu.json --device cuda
.venv/bin/python generate.py --prompt "O Brasil possui uma grande diversidade"
.venv/bin/python chat.py
.venv/bin/python -m pytest tests/ -q
```

## 13. Testes

17 testes (`tokenizer, dataset, attention+causal, model+overfit+GPU,
generation, checkpoint, chat`). GPU-skips quando sem CUDA.

## 14. Resultados

Ver `docs/RESULTS.md` (todos os números medidos, exemplos reais).

## 15. Limitações

Modelo pequeno (17M), corpus limitado vs. LLMs comerciais, contexto 256,
alucinações e incoerências possíveis, sem conhecimento atualizado,
sem memória persistente. Detalhes em `docs/RESULTS.md`.

## 16. Estrutura do projeto

```
train.py generate.py chat.py evaluate.py | configs/ | scripts/
src/ (tokenizer, dataset, attention, model, training, generation, chat, utils)
tests/ | eval/prompts_ptbr.txt | docs/ | metadata/ | experiments/ | logs/
artifacts/ (tokenizer, checkpoints, evaluation) | data/ (raw, processed)
```

## 17. Licença

Código: MIT (ver LICENSE, a criar na publicação). Dados: cada fonte mantém
sua licença (Wikipedia CC BY-SA; QA-PT verificar; seed próprio livre) —
detalhes em `docs/DATASET_SELECTION.md`. Este repo não redistribui corpora.
