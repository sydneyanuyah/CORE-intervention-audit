#!/usr/bin/env python3
"""Freeze the Llama-3.3-70B F3 real/placebo experiment."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path("${PRIVATE_STORAGE_ROOT}/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/not-published")
SOURCE=ROOT/"data/source_runs/f3"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 graphs=[]
 for graph_seed in range(3081,3101):
  directory=SOURCE/f"graph_{graph_seed}"
  files={name:{"path":str(directory/name),"sha256":sha(directory/name)} for name in ("graph.json","worlds.json","records_real.json","records_placebo.json","placebo_spec.json")}
  graphs.append({"graph_seed":graph_seed,"files":files})
 cells=[]
 for seed,graph_seed in zip(range(601,621),range(3081,3101)):
  for condition in ("real","placebo"):
   for method in ("o2","o3"):
    cells.append({"cell_id":f"llama70b-f3:{condition}:{method}:{seed}","condition":condition,"method":method,"seed":seed,"graph_seed":graph_seed,"output":f"outputs/llama70b-f3/{condition}/{method}/seed-{seed}","gpu_count":1})
 manifest={"protocol":"llama3_3_70b_f3_causal_placebo_v1","status":"preregistered_prelaunch","model":"meta-llama/Llama-3.3-70B-Instruct","model_revision":MODEL.name,"model_snapshot":str(MODEL),"model_config_sha256":sha(MODEL/"config.json"),"feature_contract":{"backbone_frozen":True,"quantization":"BF16 frozen feature extraction","pooling":"last non-padding final-layer hidden state","max_length":512},"operator_contract":{"methods":["o2","o3"],"rank":16,"nodes":[f"X{i:02d}" for i in range(30)]},"training_contract":{"steps":1200,"batch_size":64,"learning_rate":0.0003,"selection":"final preregistered step"},"graphs":graphs,"seeds":list(range(601,621)),"conditions":["real","placebo"],"cells":cells,"feature_cache":"outputs/llama70b-f3/cache/xor_features.pt","feature_cache_summary":"outputs/llama70b-f3/cache/cache_summary.json","allowed_splits":["train","validation"],"evaluation_split":"validation","test_evaluated":False}
 out=ROOT/"registry/llama70b_f3_manifest.json";out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"cells":len(cells),"manifest_sha256":sha(out)}))
if __name__=="__main__":main()
