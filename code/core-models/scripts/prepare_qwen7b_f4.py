#!/usr/bin/env python3
"""Freeze the Qwen2.5-7B arm of F4 reader-scale evaluation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];MODEL=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/not-published";SOURCE=Path("${PRIVATE_STORAGE_ROOT}/CORE_f4/runs")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 graphs=[]
 for graph_seed in range(3101,3126):
  d=SOURCE/f"graph_{graph_seed}";files={n:{"path":str(d/n),"sha256":sha(d/n)} for n in ("graph.json","worlds.json","records_real.json")};graphs.append({"graph_seed":graph_seed,"files":files})
 cells=[{"cell_id":f"qwen7b-f4:7b:{method}:{seed}","condition":"real","method":method,"seed":seed,"graph_seed":seed+2400,"output":f"outputs/qwen7b-f4/7b/{method}/seed-{seed}","gpu_count":1} for seed in range(701,726) for method in ("o2","o3")]
 m={"protocol":"qwen2p5_7b_f4_reader_scale_arm_v1","status":"preregistered_prelaunch","model":"Qwen/Qwen2.5-7B-Instruct","model_revision":MODEL.name,"model_snapshot":str(MODEL),"model_config_sha256":sha(MODEL/"config.json"),"feature_contract":{"backbone_frozen":True,"quantization":"NF4 feature extraction only","pooling":"last non-padding final-layer hidden state","max_length":512,"scale_arm":"7b"},"operator_contract":{"methods":["o2","o3"],"rank":16,"nodes":[f"X{i:02d}" for i in range(30)]},"training_contract":{"steps":1200,"batch_size":64,"learning_rate":0.0003,"selection":"final preregistered step"},"graphs":graphs,"seeds":list(range(701,726)),"conditions":["real"],"cells":cells,"feature_cache":"outputs/qwen7b-f4/cache/xor_features.pt","feature_cache_summary":"outputs/qwen7b-f4/cache/cache_summary.json","allowed_splits":["train","validation"],"evaluation_split":"validation","test_evaluated":False}
 out=ROOT/"registry/qwen7b_f4_manifest.json";out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(json.dumps({"cells":len(cells),"manifest_sha256":sha(out)}))
if __name__=="__main__":main()
