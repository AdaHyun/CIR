#!/usr/bin/env bash
# User-run only, no sudo, no server connection or GPU selection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?Choose --cpu or --cu121 explicitly}"
case "$MODE" in --cpu) INDEX=https://download.pytorch.org/whl/cpu;; --cu121) INDEX=https://download.pytorch.org/whl/cu121;; *) echo 'Use --cpu or --cu121'; exit 2;; esac
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10-3.12 for the pinned wheel set"'
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv || { echo 'venv unavailable; see docs/A100_RUNBOOK.md for no-sudo virtualenv/uv alternatives'; exit 1; }
fi
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
mkdir -p reports/environment
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url "$INDEX"
.venv/bin/python -m pip install -r requirements-full.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > "reports/environment/freeze-$(date -u +%Y%m%dT%H%M%SZ).txt"
.venv/bin/python -c 'import torch,torchvision,open_clip,transformers; print("torch",torch.__version__,"torchvision",torchvision.__version__,"CUDA wheel runtime",torch.version.cuda,"CUDA available",torch.cuda.is_available())'
echo 'Environment ready. Activate with: source .venv/bin/activate'
