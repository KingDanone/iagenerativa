#!/usr/bin/env bash
# Demo (sem treino longo): versões, GPU, testes, geração, avaliação.
set -u
PY=.venv/bin/python
echo "== Python =="; $PY --version
echo "== PyTorch/GPU =="; $PY -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "== Testes =="; .venv/bin/python -m pytest tests/ -q 2>&1 | tail -2
echo "== Checkpoint =="; ls -lh artifacts/checkpoints/ artifacts/tokenizer/
echo "== Geração (3 prompts) =="
$PY generate.py --prompt "O Brasil possui uma grande diversidade" --max-new-tokens 60 2>&1 | tail -3
$PY generate.py --prompt "Python é uma linguagem" --max-new-tokens 60 2>&1 | tail -3
$PY generate.py --prompt "A inteligência artificial é uma área" --max-new-tokens 60 2>&1 | tail -3
echo "(chat interativo: .venv/bin/python chat.py — não executado aqui)"
