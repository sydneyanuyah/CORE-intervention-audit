#!/bin/bash
#SBATCH --job-name=core-qwen32-all
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/qwen32-all-%j.log

set -euo pipefail
ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs/large_model_cluster"
cd "$ROOT"
test -L .venv || ln -s ../env .venv
.venv/bin/python scripts/run_qwen32_all.py
