# Treinamento

## Dados

31,7M tokens BPE (train 30,1M / val 1,6M, 95/5), uint16 em `data/processed/`,
janela `block_size=256`, shift input→target (`src/dataset.py`).

## Otimização

- **AdamW** (`lr=3e-4`, `weight_decay=0.1`, betas 0.9/0.95).
- **Warmup linear** (200 steps) + **decaimento cosseno** até `min_lr=3e-5`
  (`src/utils.py::lr_schedule`, sem LambdaLR para ser legível).
- **Batch**: físico 8 × acumulação 8 = **efetivo 64** (16.384 tokens/step).
- **AMP fp16 + GradScaler** em CUDA (RTX 3050/Ampere prioriza fp16; CPU faz fallback fp32).
- `zero_grad(set_to_none=True)`, clip 1.0, eval com `torch.no_grad()`.

## Custo medido (RTX 3050 6GB)

| Métrica | Valor real |
|---|---|
| tokens/s | ~33.000 |
| VRAM pico | ~0,8 GB de 5,7 GB |
| 5000 steps | ~44 min (~82M tokens, ~2,7 épocas) |

Configuração justificada por medição: sobra de VRAM proposital (margem p/ batch
maior em expansões futuras); gargalo é computação, não memória.

## Checkpoints / resume

`artifacts/checkpoints/latest.pt` (sempre) e `best.pt` (menor val loss),
com `model_state`, `optimizer_state`, `step`, `config`, losses.
Resume: `python train.py --resume artifacts/checkpoints/latest.pt`.

## Debug / overfit (ordem obrigatória)

1. `pytest` (17 testes, inclui overfit em dataset minúsculo).
2. `train.py --config configs/debug.json` (~120 steps, loss deve cair).
3. `--dry-run` antes de qualquer treino longo (estima VRAM).
4. Se OOM: nesta ordem — ↓micro-batch, ↑acumulação, AMP, ↓contexto,
   checkpointing, ↓embedding, ↓layers (ver código e relatório).

## Logs

`logs/train.log` (step/loss/lr/tok-s/vram) e `logs/train_full.log` (treino principal).
