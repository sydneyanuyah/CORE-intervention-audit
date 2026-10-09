#!/bin/bash
#SBATCH --job-name=core-l3-independent
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/l3-independent-v2-%j.log

set -euo pipefail
ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs/large_model_cluster"
cd "$ROOT"
test -L .venv || ln -s ../env .venv
.venv/bin/python scripts/run_l3_independent.py --gpu-count 3 qwen32 llama70b
