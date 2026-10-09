#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(task,train,val):
 core=Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py");cells=[]
 for method in ("baseline","o3"):
  for seed in range(101,121):
   cp=ROOT/f"outputs/f1/o3/seed-{seed}/operator.pt";cells.append({"cell_id":f"{task}:{method}:seed-{seed}","method":method,"seed":seed,"checkpoint":str(cp),"checkpoint_sha256":sha(cp),"output":f"outputs/{task}/{method}/seed-{seed}","world_size":4,"test_evaluated":False})
 return {"protocol":f"{task}_frozen_f1_candidate_probe_v1","status":"preregistered_prelaunch","model":"google-bert/bert-base-uncased","world_size":4,"operator_source":str(core),"operator_source_sha256":sha(core),"train_data":str(train),"train_data_sha256":sha(train),"validation_data":str(val),"validation_data_sha256":sha(val),"methods":["baseline","o3"],"seeds":list(range(101,121)),"operator_policy":"F1 O3 checkpoint frozen; only shared candidate probe backbone/head/state tokens fit","cells":cells,"test_evaluated":False}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--task",choices=("t5","g1"));args=parser.parse_args()
 specs={"t5":(ROOT/"data/experiments/t5/native_train.jsonl",ROOT/"data/experiments/t5/native_validation.jsonl"),"g1":(ROOT/"data/experiments/g1/estimand_train.jsonl",ROOT/"data/experiments/g1/estimand_validation.jsonl")}
 for task,(tr,va) in specs.items():
  if args.task is None or args.task==task:(ROOT/f"registry/{task}_manifest.json").write_text(json.dumps(build(task,tr,va),indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
