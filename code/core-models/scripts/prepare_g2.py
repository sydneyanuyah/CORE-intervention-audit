#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 data=ROOT/"data/experiments/g2/sequences_validation.jsonl"; core=Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py"); cells=[]
 for method in ("o2","o3"):
  for seed in range(101,121):
   checkpoint=ROOT/f"outputs/f1/{method}/seed-{seed}/operator.pt"; cells.append({"cell_id":f"g2:{method}:seed-{seed}","method":method,"seed":seed,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"output":f"outputs/g2/{method}/seed-{seed}","world_size":4,"test_evaluated":False})
 manifest={"protocol":"g2_frozen_operator_last_write_wins_v1","status":"preregistered_prelaunch","validation_data":str(data),"validation_data_sha256":sha(data),"operator_source":str(core),"operator_source_sha256":sha(core),"state_probe":"fixed_nonlearned_sinusoidal_16x768_plus_binary_state_offset","primary":"hidden_agreement_at_1e-5:o3-minus-o2","cells":cells,"test_evaluated":False}
 (ROOT/"registry/g2_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
