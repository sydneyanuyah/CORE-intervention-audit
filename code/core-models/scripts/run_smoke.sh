#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python3 src/train.py \
  --data-root data/core \
  --output-dir outputs/bert-base-smoke \
  --model google-bert/bert-base-uncased \
  --epochs 1 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --max-length 256 \
  --precision bf16 \
  --max-train-samples 16 \
  --max-eval-samples 16 \
  --max-steps 1 \
  --num-workers 0
