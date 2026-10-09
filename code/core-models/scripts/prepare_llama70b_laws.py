#!/usr/bin/env python3
"""Register Llama-3.3-70B frozen-feature L1/L2/L3 law cells."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=ROOT/"registry/llama70b_f1_manifest.json";amend=ROOT/"registry/llama70b_f1_cache_amendment.json"
 if not source.is_file() or not amend.is_file():raise FileNotFoundError("frozen Phi F1 source missing")
 specs={
  "l1":(["core_base_1law","core_i_3law","core_full_4law","do_nothing","do_everything","random_init"],range(901,921)),
  "l2":(["full","drop_identity","drop_idempotence","drop_commutation","drop_selective_invariance"],range(921,941)),
  "l3":(["prompting","lora_matched","loreft","task_vector_add","router","o1","o2","o3"],range(941,946)),
 }
 for task,(methods,seeds) in specs.items():
  cells=[{"cell_id":f"llama70b-{task}:{method}:{seed}","method":method,"seed":seed,"source_seed":501+(seed-min(seeds))%20,"output":f"outputs/llama70b-{task}/{method}/seed-{seed}","gpu_count":1} for seed in seeds for method in methods]
  m={"protocol":f"llama3_3_70b_{task}_law_probe_v1","task":task,"source_manifest":str(source),"source_manifest_sha256":sha(source),"source_amendment":str(amend),"source_amendment_sha256":sha(amend),"methods":methods,"seeds":list(seeds),"cells":cells,"training_steps":600,"law_weight":0.1,"evaluation_split":"validation","backbone_frozen":True,"test_evaluated":False}
  p=ROOT/f"registry/llama70b_{task}_manifest.json";p.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(task,len(cells),sha(p))
if __name__=="__main__":main()
