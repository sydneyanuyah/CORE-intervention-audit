#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
torchrun --standalone --nproc_per_node=4 src/train.py \
  --data-root data/core \
  --output-dir outputs/bert-base-uncased-seed-20260904 \
  --model google-bert/bert-base-uncased \
  --epochs 3 \
  --train-batch-size 32 \
  --eval-batch-size 64 \
  --max-length 384 \
  --precision bf16 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed 20260904 \
  --num-workers 2
