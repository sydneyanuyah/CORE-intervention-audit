#!/bin/bash
#SBATCH --job-name=core-llama70b-all
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/llama70b-all-%j.log

set -euo pipefail
ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs/large_model_cluster"
cd "$ROOT"
test -L .venv || ln -s ../env .venv
.venv/bin/python scripts/run_llama70b_all.py
