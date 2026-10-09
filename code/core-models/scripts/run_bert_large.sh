#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
GPU_COUNT="${GPU_COUNT:-8}"
if [[ "$GPU_COUNT" != "8" && "$GPU_COUNT" != "16" ]]; then
  echo "BERT-large requires GPU_COUNT=8 or GPU_COUNT=16" >&2
  exit 2
fi
torchrun --standalone --nproc_per_node="$GPU_COUNT" src/train.py \
  --data-root data/core \
  --output-dir outputs/bert-large-uncased-epoch1-seed-20260904 \
  --model google-bert/bert-large-uncased \
  --epochs 1 \
  --train-batch-size 12 \
  --eval-batch-size 24 \
  --max-length 384 \
  --precision bf16 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed 20260904 \
  --num-workers 2
