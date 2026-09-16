# Roteiro de vídeo demo (5–8 min)

## 00:00–00:30 — Apresentação
"Este projeto implementa e treina do zero um pequeno modelo de linguagem
autoregressivo em português: tokenizer BPE próprio, Transformer decoder-only
de 17,4M parâmetros, 31,7M tokens de treino, tudo local numa RTX 3050 6GB."

## 00:30–01:30 — Arquitetura
Mostrar `src/model.py` (MiniGPT), diagrama do README, `docs/ARCHITECTURE.md`.
Destacar: atenção causal manual, máscara triangular, Pre-LN + residual, LM head.

## 01:30–02:30 — Dataset + tokenizer
`docs/DATASET_SELECTION.md`, `metadata/dataset_manifest.json`,
`scripts/` (download→clean→tokenizer→build).
Demo: `scripts/train_tokenizer.py` roundtrip "São Luís é uma cidade brasileira."

## 02:30–03:30 — Treinamento
`logs/train_full.log`: loss 8.3→~, tokens/s ~33k, VRAM 0,8GB.
`configs/gpu.json`, checkpoints `latest.pt`/`best.pt`, resume.

## 03:30–04:30 — Geração
`python generate.py --prompt "O Brasil possui uma grande diversidade"`.
Mostrar 2–3 prompts de `eval/prompts_ptbr.txt` com saídas REAIS
(mesmo que imperfeitas — não editar).

## 04:30–06:00 — Chat ao vivo
```
Você: Olá
Você: Meu nome é João.
Você: Qual é o meu nome?
/history /stats /reset /exit
```
Se errar o nome: apontar a janela de contexto (docs/CHAT.md) como causa —
mostrar comportamento real, sem falsificar.

## 06:00–07:00 — Testes e resultados
`pytest` (17 passed), `docs/RESULTS.md` (números reais), limitações,
estrutura do repo e `.gitignore` (sem datasets/checkpoints no Git).
