#!/usr/bin/env python3
"""Fit one layer-specific O3 head for Qwen G3."""
import argparse,hashlib,json
from pathlib import Path
import torch
from core_7b.operator_probe import O3StateGated,StateHead,balanced_loss
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--layer",type=int,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256:raise ValueError("manifest mismatch")
 m=json.loads(a.manifest.read_text());source=json.loads(Path(m["source_manifest"]).read_text());parts=[torch.load(f"outputs/qwen7b-g3/cache/shard-{i:02d}.pt",map_location="cpu",weights_only=False) for i in range(32)];lookup={}
 for q in parts:
  if q["manifest_sha256"]!=a.manifest_sha256:raise ValueError("cache mismatch")
  for r,v in zip(q["rows"],q["features"][a.layer]):lookup[(r["graph_seed"],r["world_id"])]=v.float()
 nodes=source["operator_contract"]["nodes"];packed=[]
 for g in source["graphs"]:
  worlds={w["world_id"]:w for w in json.loads(Path(g["files"]["worlds.json"]["path"]).read_text())}
  for r in json.loads(Path(g["files"]["records_real.json"]["path"]).read_text()):packed.append((r["split"],lookup[(g["graph_seed"],r["world_id"])],nodes.index(r["intervention"]["target"])*2+int(r["intervention"]["value"]),[int(r["intervened_state"][n]) for n in nodes],[int(worlds[r["world_id"]]["factual_state"][n]) for n in nodes]))
 dev=torch.device("cuda:0");torch.manual_seed(900);h=packed[0][1].numel();op=O3StateGated(60,h,16).to(dev);head=StateHead(h,30).to(dev);params=list(op.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4)
 def batch(split):
  q=[r for r in packed if r[0]==split];return torch.stack([r[1] for r in q]).to(dev),torch.tensor([r[2] for r in q],device=dev),torch.tensor([r[3] for r in q],device=dev),torch.tensor([r[4] for r in q],device=dev)
 x,i,y,f=batch("train");gen=torch.Generator().manual_seed(900)
 for step in range(600):
  ix=torch.randint(0,len(x),(256,),generator=gen).to(dev);loss=balanced_loss(head(op(x[ix],i[ix])),y[ix],y[ix]!=f[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 with torch.no_grad():x,i,y,f=batch("validation");pred=head(op(x,i)).argmax(-1);ch=y!=f;score=.5*((pred[ch]==y[ch]).float().mean()+(pred[~ch]==y[~ch]).float().mean())
 out=Path(f"outputs/qwen7b-g3/layer-{a.layer}");out.mkdir(parents=True,exist_ok=False);s={"protocol":m["protocol"],"layer":a.layer,"validation_balanced_accuracy":float(score),"seed":900,"manifest_sha256":a.manifest_sha256,"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
