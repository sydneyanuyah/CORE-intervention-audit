#!/usr/bin/env bash
set -euo pipefail

cd ${CORE_PROJECT_ROOT}
mkdir -p logs/f2-baselines
exec 9>logs/f2-baselines/lora-tail-dispatcher.lock
if ! flock -n 9; then
  echo "another F2 LoRA tail dispatcher holds the lock" >&2
  exit 2
fi

current=(309 310 311 312 313 314 315 316)
queue=(317 318 319 320)
queue_index=0

timestamp() { date -Is; }

validate_summary() {
  local seed=$1
  local summary="outputs/f2/xor/lora_matched/seed-${seed}/run_summary.json"
  local operator="outputs/f2/xor/lora_matched/seed-${seed}/operator.pt"
  [[ -f "$summary" && -f "$operator" ]] || return 1
  .venv/bin/python - "$summary" "$seed" <<'PY'
import json
import sys

path, expected_seed = sys.argv[1], int(sys.argv[2])
payload = json.load(open(path, encoding="utf-8"))
assert payload["protocol"] == "f2_cell_v1"
assert payload["method"] == "lora_matched"
assert payload["seed"] == expected_seed
assert payload["graph_seed"] == expected_seed + 2760
assert payload["world_size"] == 4
assert payload["test_evaluated"] is False
assert payload["manifest_sha256"] == "not-published"
assert payload["cells_sha256"] == "not-published"
assert payload["amendment_sha256"] == "not-published"
assert payload["lora_rank"] == 34
assert payload["trainable_parameter_count"] == 1253376
PY
}

is_running() {
  local seed=$1
  pgrep -f "src/train_f2_baselines.py --method lora_matched --seed ${seed} " >/dev/null
}

echo "$(timestamp) dispatcher_started current=${current[*]} queue=${queue[*]}"
while true; do
  occupied=0
  for group in 0 1 2 3 4 5 6 7; do
    seed=${current[$group]}
    [[ -n "$seed" ]] || continue
    occupied=$((occupied + 1))
    if validate_summary "$seed"; then
      echo "$(timestamp) finalized seed=$seed group=$group"
      if (( queue_index < ${#queue[@]} )); then
        next=${queue[$queue_index]}
        queue_index=$((queue_index + 1))
        echo "$(timestamp) launching seed=$next group=$group after=$seed"
        F2_GPU_GROUP_START=$group scripts/launch_f2_baseline_wave.sh lora_matched "$next"
        current[$group]=$next
      else
        current[$group]=""
      fi
    elif ! is_running "$seed"; then
      echo "$(timestamp) FAILED seed=$seed group=$group stopped_without_valid_summary" >&2
      exit 1
    fi
  done
  if (( occupied == 0 )); then
    echo "$(timestamp) dispatcher_complete"
    exit 0
  fi
  sleep 5
done
