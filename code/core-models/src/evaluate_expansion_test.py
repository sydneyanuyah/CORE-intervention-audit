#!/usr/bin/env python3
"""Score every frozen expansion checkpoint on its registered held-out cache."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from core_7b.operator_probe import O3StateGated

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class Head(nn.Module):
 def __init__(self,h,o): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,o))
 def forward(self,x): return self.net(x)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--manifest-sha256",required=True); p.add_argument("--cell-id",required=True); a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256: raise ValueError("manifest mismatch")
 m=json.loads(a.manifest.read_text())
 if m.get("authorization")!="EXPANSION_FINAL_ONE_SHOT" or m.get("test_evaluated") is not True: raise PermissionError("not authorized")
 cells=[cell for cell in m["source_cells"] if cell["cell_id"]==a.cell_id]
 if len(cells)!=1: raise ValueError("unregistered source cell")
 cell=cells[0]; checkpoint=Path(cell["checkpoint"])
 if sha(checkpoint)!=cell["checkpoint_sha256"]: raise ValueError("checkpoint changed")
 training=Path(m["training_manifest"])
 if sha(training)!=m["training_manifest_sha256"]: raise ValueError("training manifest changed")
 rows=[json.loads(line) for line in Path(m["data"]).read_text().splitlines() if line.strip()]
 parts=[torch.load(f"outputs/{m['model_key']}-{m['task']}-final-test/cache/shard-{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(m["feature_shards"])]
 hidden=parts[0]["features"].shape[1]; x=torch.empty((len(rows),hidden),dtype=torch.float32)
 for part in parts:
  if part.get("manifest_sha256")!=a.manifest_sha256 or part.get("test_evaluated") is not True: raise ValueError("cache mismatch")
  for index,value in zip(part["indices"],part["features"]):
   if index<len(rows): x[index]=value.float()
 payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
 if payload.get("test_evaluated") is not False: raise ValueError("source checkpoint is not test-clean")
 outdim=payload["head"]["net.1.weight"].shape[0]; method=cell["method"]
 op=O3StateGated(256,hidden,16) if method=="o3" else nn.Identity(); head=Head(hidden,outdim); op.load_state_dict(payload["operator"]); head.load_state_dict(payload["head"]); op.eval(); head.eval()
 command=torch.tensor([int(row["command"])%256 for row in rows])
 with torch.no_grad(): logits=head(op(x,command)) if method=="o3" else head(x)
 task=m["task"]
 if task in {"t2","t4"}:
  width=outdim//3; labels=torch.full((len(rows),width),-100,dtype=torch.long)
  for index,row in enumerate(rows):
   if len(row["label"])>width: raise ValueError("test graph exceeds trained output width")
   labels[index,:len(row["label"])]=torch.tensor(row["label"])
  pred=logits.reshape(len(rows),width,3).argmax(2); valid=labels!=-100
  accuracy=float(((pred==labels)|~valid).all(1).float().mean()); detail={"variable_accuracy":float((pred[valid]==labels[valid]).float().mean()),"maximum_graph_width":width}
  by_source={}
  for source in sorted({row["source"] for row in rows}):
   idx=torch.tensor([i for i,row in enumerate(rows) if row["source"]==source]); mask=valid[idx]
   by_source[source]={"rows":len(idx),"exact_accuracy":float((((pred[idx]==labels[idx])|~mask).all(1)).float().mean()),"variable_accuracy":float((pred[idx][mask]==labels[idx][mask]).float().mean())}
  detail["by_source"]=by_source
 else:
  labels=torch.tensor([row["label"] for row in rows]); pred=(logits[:,0]>0).long(); groups=defaultdict(list)
  for index,row in enumerate(rows): groups[row["item_id"]].append((float(logits[index,0]),int(labels[index]),row["source"]))
  accuracy=sum(max(values)[1] for values in groups.values())/len(groups); detail={"candidate_accuracy":float((pred==labels).float().mean()),"items":len(groups)}
  detail["by_source"]={source:{"items":len([v for v in groups.values() if v[0][2]==source]),"accuracy":sum(max(v)[1] for v in groups.values() if v[0][2]==source)/len([v for v in groups.values() if v[0][2]==source])} for source in sorted({r["source"] for r in rows})}
 output=Path(f"outputs/{m['model_key']}-{task}-final-test/{method}/seed-{cell['seed']}"); output.mkdir(parents=True,exist_ok=False)
 result={"protocol":m["protocol"],"cell_id":a.cell_id,"method":method,"seed":cell["seed"],"test_accuracy":accuracy,**detail,"checkpoint_sha256":cell["checkpoint_sha256"],"manifest_sha256":a.manifest_sha256,"test_rows":m["test_rows"],"evaluation_split":"test","test_evaluated":True,"excluded_sources":["com2"]}
 (output/"test_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result))

if __name__=="__main__": main()
