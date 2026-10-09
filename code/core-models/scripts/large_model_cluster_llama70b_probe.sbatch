#!/bin/bash
#SBATCH --job-name=core-llama70b-probe
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=00:45:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/llama70b-probe-%j.log

set -euo pipefail

ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs/large_model_cluster"
cd "$ROOT"
${PRIVATE_STORAGE_ROOT}/env/bin/python scripts/probe_kailash_llama70b.py
