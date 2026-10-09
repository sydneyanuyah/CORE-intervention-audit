#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 9 ]]; then
  echo "usage: $0 {prompting|lora_matched} SEED [SEED ...]" >&2
  exit 2
fi
method=$1
shift
if [[ "$method" != prompting && "$method" != lora_matched ]]; then
  echo "unregistered F2 baseline method: $method" >&2
  exit 2
fi

manifest_sha=not-published
cells_sha=not-published
amendment_sha=not-published
core_sha=not-published

[[ $(sha256sum registry/f2_manifest.json | cut -d' ' -f1) == "$manifest_sha" ]]
[[ $(sha256sum registry/f2_cells.json | cut -d' ' -f1) == "$cells_sha" ]]
[[ $(sha256sum registry/f2_baseline_execution_amendment.json | cut -d' ' -f1) == "$amendment_sha" ]]
mkdir -p logs/f2-baselines

group=${F2_GPU_GROUP_START:-0}
if (( group < 0 || group > 7 || group + $# > 8 )); then
  echo "requested cells do not fit in GPU groups 0-7" >&2
  exit 2
fi
for seed in "$@"; do
  if (( seed < 301 || seed > 320 )); then
    echo "unregistered F2 seed: $seed" >&2
    exit 2
  fi
  graph_seed=$((seed + 2760))
  artifact_dir="${PRIVATE_STORAGE_ROOT}/CORE_f2/runs/graph_${graph_seed}"
  reader="outputs/f2/readers/xor/seed-${seed}/reader.pt"
  output="outputs/f2/xor/${method}/seed-${seed}"
  [[ -f "$reader" && -f "$artifact_dir/graph.json" ]]
  [[ ! -e "$output" ]]
  first=$((group * 4))
  gpu_set="$first,$((first + 1)),$((first + 2)),$((first + 3))"
  log="logs/f2-baselines/xor-${method}-${seed}.log"
  CUDA_VISIBLE_DEVICES="$gpu_set" nohup .venv/bin/torchrun --standalone --nproc_per_node=4 \
    src/train_f2_baselines.py \
    --method "$method" --seed "$seed" --graph-seed "$graph_seed" \
    --artifact-dir "$artifact_dir" --reader-checkpoint "$reader" --output-dir "$output" \
    --core-components ${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py \
    --expected-core-sha256 "$core_sha" \
    --manifest registry/f2_manifest.json --expected-manifest-sha256 "$manifest_sha" \
    --cells registry/f2_cells.json --expected-cells-sha256 "$cells_sha" \
    --amendment registry/f2_baseline_execution_amendment.json \
    --expected-amendment-sha256 "$amendment_sha" \
    >"$log" 2>&1 < /dev/null &
  echo "$method seed=$seed graph_seed=$graph_seed gpus=$gpu_set pid=$! log=$log"
  group=$((group + 1))
done
