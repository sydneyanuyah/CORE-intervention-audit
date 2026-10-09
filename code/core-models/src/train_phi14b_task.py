#!/usr/bin/env python3
"""Train one registered frozen-feature baseline/O3 task cell."""
import argparse,hashlib,json,random
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from core_7b.operator_probe import O3StateGated
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Head(nn.Module):
 def __init__(self,h,o):super().__init__();self.net=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,o))
 def forward(self,x):return self.net(x)
def main():
 p=argparse.ArgumentParser();p.add_argument("--cell-id",required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256:raise ValueError("manifest mismatch")
 m=json.loads(a.manifest.read_text());cell=[c for c in m["cells"] if c["cell_id"]==a.cell_id]
 if len(cell)!=1 or m["test_evaluated"] is not False:raise ValueError("unregistered cell")
 cell=cell[0];rows=[json.loads(x) for x in Path(m["data"]).read_text().splitlines() if x.strip()];parts=[torch.load(f"outputs/phi14b-{m['task']}/cache/shard-{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(m["feature_shards"])];h=parts[0]["features"].shape[1];x=torch.empty((len(rows),h),dtype=torch.float32)
 for q in parts:
  if q["manifest_sha256"]!=a.manifest_sha256 or q["test_evaluated"] is not False:raise ValueError("cache mismatch")
  for i,v in zip(q["indices"],q["features"]):
   if i<len(rows):x[i]=v.float()
 task=m["task"];labels=[r["label"] for r in rows]
 if task in {"t2","t4"}:width=max(map(len,labels));outdim=width*3;kind="direction"
 elif task=="g2":outdim=8;kind="bits"
 elif task in {"t3","t5"}:outdim=1;kind="binary"
 else:outdim=max(labels)+1;kind="class"
 dev=torch.device("cuda:0");seed=cell["seed"];random.seed(seed);torch.manual_seed(seed);x=x.to(dev);cmd=torch.tensor([int(r["command"])%256 for r in rows],device=dev);train=torch.tensor([r["split"]=="train" for r in rows],device=dev);val=~train
 op=O3StateGated(256,h,16).to(dev) if cell["method"]=="o3" else nn.Identity().to(dev);head=Head(h,outdim).to(dev);params=list(op.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4)
 def forward(ix):return head(op(x[ix],cmd[ix])) if cell["method"]=="o3" else head(x[ix])
 if kind=="direction":
  y=torch.full((len(labels),width),-100,dtype=torch.long,device=dev)
  for row,label in enumerate(labels):y[row,:len(label)]=torch.tensor(label,device=dev)
 else:y=torch.tensor(labels,device=dev,dtype=torch.float if kind in {"binary","bits"} else torch.long)
 gen=torch.Generator().manual_seed(seed);ti=train.nonzero().flatten()
 for step in range(600):
  ix=ti[torch.randint(0,len(ti),(min(256,len(ti)),),generator=gen).to(dev)];z=forward(ix)
  if kind=="direction":loss=nn.functional.cross_entropy(z.reshape(len(ix),width,3).reshape(-1,3),y[ix].reshape(-1),ignore_index=-100)
  elif kind=="bits":loss=nn.functional.binary_cross_entropy_with_logits(z,y[ix])
  elif kind=="binary":loss=nn.functional.binary_cross_entropy_with_logits(z[:,0],y[ix])
  else:loss=nn.functional.cross_entropy(z,y[ix])
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 with torch.no_grad():
  ix=val.nonzero().flatten();z=forward(ix)
  if kind=="direction":
   pred=z.reshape(len(ix),width,3).argmax(2);valid=y[ix]!=-100;accuracy=float(((pred==y[ix])|~valid).all(1).float().mean());detail={"variable_accuracy":float((pred[valid]==y[ix][valid]).float().mean()),"maximum_graph_width":width}
  elif kind=="bits":pred=(z>0).long();accuracy=float((pred==y[ix].long()).all(1).float().mean());detail={"bit_accuracy":float((pred==y[ix]).float().mean())}
  elif kind=="binary":
   pred=(z[:,0]>0).long();accuracy=float((pred==y[ix].long()).float().mean());detail={}
   if task=="t5":
    groups=defaultdict(list)
    for local,j in enumerate(ix.tolist()):groups[rows[j]["item_id"]].append((float(z[local,0]),int(y[j])))
    accuracy=sum(max(v)[1] for v in groups.values())/len(groups);detail={"candidate_accuracy":float((pred==y[ix].long()).float().mean()),"items":len(groups)}
   if task=="t3":
    by=defaultdict(dict)
    for local,j in enumerate(ix.tolist()):by[rows[j]["pair_id"]][rows[j]["variant"]]=int(pred[local])
    complete=[v for v in by.values() if len(v)==3];detail={"pairs":len(complete),"paraphrase_movement":sum(v["clean"]!=v["paraphrase"] for v in complete)/len(complete),"refactorization_movement":sum(v["clean"]!=v["refactorization"] for v in complete)/len(complete)}
  else:pred=z.argmax(1);accuracy=float((pred==y[ix]).float().mean());detail={}
 out=Path(cell["output"]);out.mkdir(parents=True,exist_ok=False);ck=out/"model.pt";torch.save({"operator":op.state_dict(),"head":head.state_dict(),"test_evaluated":False},ck);summary={"protocol":m["protocol"],"cell_id":cell["cell_id"],"task":task,"method":cell["method"],"seed":seed,"validation_accuracy":accuracy,**detail,"manifest_sha256":a.manifest_sha256,"checkpoint_sha256":sha(ck),"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary))
if __name__=="__main__":main()
