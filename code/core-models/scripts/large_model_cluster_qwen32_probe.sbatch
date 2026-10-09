#!/bin/bash
#SBATCH --job-name=core-qwen32-probe
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/qwen32-probe-%j.log

set -euo pipefail

ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs/large_model_cluster"
cd "$ROOT"
${PRIVATE_STORAGE_ROOT}/env/bin/python scripts/probe_kailash_qwen32.py
