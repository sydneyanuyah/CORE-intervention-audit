#!/usr/bin/env python3
"""Register the Qwen2.5-32B F1 XOR O2/O3 extension."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path("${PRIVATE_STORAGE_ROOT}/hf_cache/models--Qwen--Qwen2.5-32B-Instruct/snapshots/not-published")
SOURCE=ROOT/"data/source_runs/f1"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 graphs=list(range(3041,3061)); entries=[]
 for g in graphs:
  d=SOURCE/f"graph_{g}"; pair=ROOT/f"data/two_edit/f1/xor_graph_{g}.jsonl"
  files={n:{"path":str(d/n),"sha256":sha(d/n)} for n in ("graph.json","worlds.json","records_real.json")}
  files["pairs"]={"path":str(pair),"sha256":sha(pair)}; entries.append({"graph_seed":g,"files":files})
 cells=[{"cell_id":f"qwen32-f1:xor:{m}:{s}","method":m,"seed":s,"graph_seed":3041+s-501,"output":f"outputs/qwen32-f1/xor/{m}/seed-{s}","gpu_count":1} for s in range(501,521) for m in ("o2","o3")]
 x={"protocol":"qwen2_5_32b_f1_xor_o2_o3_v1","status":"preregistered_prelaunch","model":"Qwen/Qwen2.5-32B-Instruct","model_revision":MODEL.name,"model_snapshot":str(MODEL),"model_config_sha256":sha(MODEL/"config.json"),"feature_contract":{"backbone_frozen":True,"quantization":"BF16 frozen feature extraction","pooling":"last non-padding final-layer hidden state","max_length":512},"operator_contract":{"methods":["o2","o3"],"rank":16,"nodes":[f"X{i:02d}" for i in range(30)]},"training_contract":{"steps":1200,"batch_size":64,"learning_rate":0.0003,"selection":"final preregistered step"},"graphs":entries,"seeds":list(range(501,521)),"cells":cells,"feature_cache":"outputs/qwen32-f1/cache/xor_features.pt","feature_cache_summary":"outputs/qwen32-f1/cache/cache_summary.json","allowed_splits":["train","validation"],"evaluation_split":"validation","test_evaluated":False}
 out=ROOT/"registry/qwen32_f1_manifest.json";out.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");print(json.dumps({"cells":len(cells),"manifest_sha256":sha(out)}))
if __name__=="__main__": main()
