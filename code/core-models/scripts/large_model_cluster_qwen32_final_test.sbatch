#!/bin/bash
#SBATCH --job-name=core-qwen32-test
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/qwen32-final-test-%j.log
set -euo pipefail
ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"; test -L .venv || ln -s ../env .venv
.venv/bin/python scripts/run_expansion_final_test_kailash.py --model-key qwen32
