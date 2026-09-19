# Guará — Marca do modelo

Guará é a família de modelos de linguagem generativa em português
treinados do zero neste projeto (Transformer decoder-only próprio,
tokenizer BPE próprio, sem pesos pré-treinados, sem API externa).

## Nomenclatura

```
Guará <versão>
```

- **Guará** = nome da marca (fixo)
- **versão** = semver do modelo (X.Y.Z ou X.Y)
  - `1.0` = primeira versão pública (atual)
  - `1.x` = refinamentos do mesmo tamanho (receitas, dados)
  - `2.x` = mudanças estruturais (maior modelo, arquitetura, etc.)

## Versões

| Versão | Params | Dados | Tipo | Checkpoint |
|--------|--------|-------|------|------------|
| Guará 1.0 | 17,4M | 49M tokens conversacionais (mix QA-PT/Aya/OASST + replay) | Ajuste fino | `artifacts/checkpoints_chat/best.pt` |
| Guará 1.0-base | 17,4M | 82M tokens gerais (Fase 2) | Baseline | `artifacts/checkpoints/best.pt` |

## Como funciona

- Um modelo grande (Qwen 3.8, Claude Opus 5, GPT-4) é treinado por etapas,
  cada uma com nome/versão pública.
- Guará segue o mesmo princípio, porém em escala de projeto:
  a versão só avança quando há mudança significativa (dados, arquitetura,
  escala). Versões pontuais (1.0.1, 1.0.2) são correções ou testes.

## Regras para bump de versão

- **1.0 → 1.1**: mais dados, mais steps, mesma arquitetura.
- **1.x → 2.0**: mudança de arquitetura (maior d/n_layer), mudança de vocab
  (tokenizer diferente), ou mudança de formato de treino (ex.: SFT explícito).
- **Tag no checkpoint**: cada checkpoint leva `model_name` + `version` no payload,
  visível via `python -c "import torch; c=torch.load('...'); print(c['model_name'], c['version'])"`.

## Rodar uma versão específica

```bash
# versão atual (1.0 conversacional) — padrão do chat.py
python chat.py

# versão base (1.0 baseline)
python chat.py --checkpoint artifacts/checkpoints/best.pt
```
