#!/bin/bash
#SBATCH --job-name=core-llama70-test
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=${CORE_PROJECT_ROOT}/logs/large_model_cluster/llama70-final-test-%j.log
set -euo pipefail
ROOT=${CORE_PROJECT_ROOT}
export HF_HOME=${PRIVATE_STORAGE_ROOT}/hf_cache TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"; test -L .venv || ln -s ../env .venv
.venv/bin/python scripts/run_expansion_final_test_kailash.py --model-key llama70b
