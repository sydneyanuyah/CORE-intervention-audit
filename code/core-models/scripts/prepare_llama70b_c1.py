#!/usr/bin/env python3
"""Register the missing Llama-3.3-70B C1 shuffled-control arm."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/"data/source_runs/f3"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 graphs=[]
 for graph_seed in range(3081,3101):
  d=SOURCE/f"graph_{graph_seed}";shuffled=ROOT/f"data/experiments/c1/shuffled/graph_{graph_seed}.jsonl";files={n:{"path":str(d/n),"sha256":sha(d/n)} for n in ("graph.json","worlds.json")};files["records_shuffled.json"]={"path":str(shuffled),"sha256":sha(shuffled)};graphs.append({"graph_seed":graph_seed,"files":files})
 cells=[{"cell_id":f"llama70b-c1:shuffled:o3:{seed}","condition":"shuffled","method":"o3","seed":seed,"graph_seed":seed+2480,"output":f"outputs/llama70b-c1/shuffled/o3/seed-{seed}","gpu_count":1} for seed in range(601,621)]
 f3m=ROOT/"registry/llama70b_f3_manifest.json";f3a=ROOT/"registry/llama70b_f3_cache_amendment.json";m={"protocol":"llama3_3_70b_c1_closing_controls_v1","status":"preregistered_prelaunch","model":"meta-llama/Llama-3.3-70B-Instruct","reuse":{"real_o3":"outputs/llama70b-f3/real/o3","noncausal_placebo_o3":"outputs/llama70b-f3/placebo/o3","source_manifest_sha256":sha(f3m)},"feature_contract":{"source_amendment":str(f3a),"source_amendment_sha256":sha(f3a)},"operator_contract":{"methods":["o3"],"rank":16,"nodes":[f"X{i:02d}" for i in range(30)]},"training_contract":{"steps":1200,"batch_size":64,"learning_rate":0.0003,"selection":"final preregistered step"},"graphs":graphs,"seeds":list(range(601,621)),"conditions":["real","noncausal_placebo","shuffled"],"cells":cells,"evaluation_split":"validation","test_evaluated":False}
 out=ROOT/"registry/llama70b_c1_manifest.json";out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(json.dumps({"new_cells":len(cells),"reused_cells":40,"manifest_sha256":sha(out)}))
if __name__=="__main__":main()
