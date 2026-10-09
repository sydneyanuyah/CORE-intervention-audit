#!/usr/bin/env python3
"""Train/evaluate one frozen-Qwen operator under a registered law condition."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from torch import nn
from core_7b.operator_probe import O2ConditionalLowRank,O3StateGated,StateHead,balanced_loss
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Add(nn.Module):
 def __init__(self,n,h):super().__init__();self.e=nn.Embedding(n,h)
 def forward(self,x,i):return x+self.e(i)
class Router(nn.Module):
 def __init__(self,n,h):super().__init__();self.e=nn.Embedding(n,h);self.g=nn.Linear(h,1)
 def forward(self,x,i):return x+torch.sigmoid(self.g(x))*self.e(i)
def laws(method):
 allx={"identity","idempotence","commutation","selective_invariance"}
 return {"core_base_1law":{"identity"},"core_i_3law":{"identity","idempotence","commutation"},"core_full_4law":allx,"full":allx,"drop_identity":allx-{"identity"},"drop_idempotence":allx-{"idempotence"},"drop_commutation":allx-{"commutation"},"drop_selective_invariance":allx-{"selective_invariance"}}.get(method,set())
def main():
 p=argparse.ArgumentParser();p.add_argument("--cell-id",required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256:raise ValueError("manifest mismatch")
 m=json.loads(a.manifest.read_text());mch=[c for c in m["cells"] if c["cell_id"]==a.cell_id]
 if len(mch)!=1 or m["test_evaluated"] is not False:raise ValueError("unregistered");c=mch[0]
 c=mch[0];sm=Path(m["source_manifest"]);am=Path(m["source_amendment"])
 if sha(sm)!=m["source_manifest_sha256"] or sha(am)!=m["source_amendment_sha256"]:raise ValueError("source changed")
 source=json.loads(sm.read_text());binding=json.loads(am.read_text());cache=Path(binding["feature_cache"])
 if sha(cache)!=binding["feature_cache_sha256"]:raise ValueError("cache changed")
 payload=torch.load(cache,map_location="cpu",weights_only=False);lookup={(r["graph_seed"],r["world_id"]):payload["features"][i].float() for i,r in enumerate(payload["records"])};nodes=source["operator_contract"]["nodes"]
 graph=next(g for g in source["graphs"] if g["graph_seed"]==next(x for x in source["cells"] if x["seed"]==c["source_seed"])["graph_seed"]);records=json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text());worlds={w["world_id"]:w for w in json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text())}
 def pack(split):
  rr=[r for r in records if r["split"]==split];x=torch.stack([lookup[(graph["graph_seed"],r["world_id"])] for r in rr]);ids=torch.tensor([nodes.index(r["intervention"]["target"])*2+int(r["intervention"]["value"]) for r in rr]);y=torch.tensor([[int(r["intervened_state"][n]) for n in nodes] for r in rr]);f=torch.tensor([[int(worlds[r["world_id"]]["factual_state"][n]) for n in nodes] for r in rr]);return x,ids,y,f
 dev=torch.device("cuda:0");random.seed(c["seed"]);torch.manual_seed(c["seed"]);tr=tuple(v.to(dev) for v in pack("train"));va=tuple(v.to(dev) for v in pack("validation"));h=payload["hidden_size"];method=c["method"]
 if method in {"do_nothing","prompting"}:op=Add(61,h).to(dev);op.e.weight.data.zero_();trainable=False
 elif method=="do_everything":op=Add(61,h).to(dev);op.e.weight.data.normal_(0,1);trainable=False
 elif method=="random_init":op=O3StateGated(61,h,16).to(dev);trainable=False
 elif method in {"lora_matched","loreft","o2"}:op=O2ConditionalLowRank(61,h,16).to(dev);trainable=True
 elif method in {"task_vector_add","o1"}:op=Add(61,h).to(dev);trainable=True
 elif method=="router":op=Router(61,h).to(dev);trainable=True
 else:op=O3StateGated(61,h,16).to(dev);trainable=True
 head=StateHead(h,30).to(dev);params=list(head.parameters())+(list(op.parameters()) if trainable else []);opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4);x,i,y,f=tr;changed=y!=f;lawset=laws(method);noop=torch.full_like(i,60)
 for step in range(m["training_steps"]):
  z=op(x,i);loss=balanced_loss(head(z),y,changed);rev=i.roll(1)
  if "identity" in lawset:loss+=m["law_weight"]*(op(x,noop)-x).square().mean()
  if "idempotence" in lawset:loss+=m["law_weight"]*(op(z,i)-z).square().mean()
  if "commutation" in lawset:loss+=m["law_weight"]*(op(op(x,i),rev)-op(op(x,rev),i)).square().mean()
  if "selective_invariance" in lawset:loss+=m["law_weight"]*(z-x).square().mean()
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 with torch.no_grad():
  x,i,y,f=va;z=op(x,i);pred=head(z).argmax(-1);ch=y!=f;pr=~ch;score=.5*((pred[ch]==y[ch]).float().mean()+(pred[pr]==y[pr]).float().mean());rev=i.roll(1);res={"identity":float((op(x,torch.full_like(i,60))-x).square().mean()),"idempotence":float((op(z,i)-z).square().mean()),"commutation":float((op(op(x,i),rev)-op(op(x,rev),i)).square().mean()),"last_write_wins":float((op(op(x,i^1),i)-op(x,i)).square().mean())}
 out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False);ck=out/"operator.pt";torch.save({"operator":op.state_dict(),"head":head.state_dict(),"test_evaluated":False},ck);s={"protocol":m["protocol"],"cell_id":c["cell_id"],"task":m["task"],"method":method,"seed":c["seed"],"source_seed":c["source_seed"],"validation_balanced_accuracy":float(score),"law_residuals":res,"enforced_laws":sorted(lawset),"manifest_sha256":a.manifest_sha256,"checkpoint_sha256":sha(ck),"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n");print(json.dumps(s))
if __name__=="__main__":main()
