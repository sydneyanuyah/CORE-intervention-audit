#!/usr/bin/env python3
"""Freeze the Qwen2.5-32B A1 addressing-path architecture screen."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];MODEL=Path("${PRIVATE_STORAGE_ROOT}/hf_cache/models--Qwen--Qwen2.5-32B-Instruct/snapshots/not-published");SOURCE=ROOT/"data/source_runs/a1"
ARMS=("t0_no_instruction","t1_text","t2a_naive","t2b_encode_only","t3a_conditioning","t3b_pointer")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 graphs=[]
 for graph_seed in range(3021,3041):
  d=SOURCE/f"graph_{graph_seed}";pair=ROOT/f"data/two_edit/confirmatory/xor_graph_{graph_seed}.jsonl";files={n:{"path":str(d/n),"sha256":sha(d/n)} for n in ("graph.json","worlds.json","records_real.json")};files["pairs"]={"path":str(pair),"sha256":sha(pair)};graphs.append({"graph_seed":graph_seed,"files":files})
 cells=[{"cell_id":f"qwen32-a1:xor:{arm}:{seed}","arm":arm,"seed":seed,"graph_seed":3021+seed-801,"output":f"outputs/qwen32-a1/xor/{arm}/seed-{seed}","gpu_count":1} for seed in range(801,821) for arm in ARMS]
 m={"protocol":"qwen2_5_32b_a1_addressing_screen_v1","status":"preregistered_prelaunch","model":"Qwen/Qwen2.5-32B-Instruct","model_revision":MODEL.name,"model_snapshot":str(MODEL),"model_config_sha256":sha(MODEL/"config.json"),"scope":"XOR development architecture screen; frozen-feature analogs are explicitly distinct from token-level BERT implementations","feature_contract":{"backbone_frozen":True,"quantization":"BF16 frozen feature extraction","pooling":"last non-padding final-layer hidden state","max_length":512},"arms":{"t0_no_instruction":"head receives factual-world feature only","t1_text":"intervention embedding added directly to reader feature","t2a_naive":"joint feature/instruction MLP without isolation","t2b_encode_only":"isolated conditional low-rank editor","t3a_conditioning":"state-gated editor","t3b_pointer":"target-pointer-weighted conditional editor with auxiliary target loss"},"operator_contract":{"rank":16,"nodes":[f"X{i:02d}" for i in range(30)]},"training_contract":{"steps":1200,"batch_size":64,"learning_rate":0.0003,"pointer_loss_weight":0.1,"selection":"final preregistered step"},"graphs":graphs,"seeds":list(range(801,821)),"cells":cells,"feature_cache":"outputs/qwen32-a1/cache/xor_features.pt","feature_cache_summary":"outputs/qwen32-a1/cache/cache_summary.json","allowed_splits":["train","validation"],"evaluation_split":"validation","test_evaluated":False}
 out=ROOT/"registry/qwen32_a1_manifest.json";out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(json.dumps({"cells":len(cells),"manifest_sha256":sha(out)}))
if __name__=="__main__":main()
