#!/usr/bin/env python3
"""Train one registered Qwen-7B A1 frozen-feature architecture cell."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from core_7b.operator_probe import O2ConditionalLowRank,O3StateGated,StateHead,balanced_loss
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Identity(nn.Module):
 def forward(self,h,i):return h
class TextAdd(nn.Module):
 def __init__(self,n,h):super().__init__();self.e=nn.Embedding(n,h)
 def forward(self,x,i):return x+self.e(i)
class JointMLP(nn.Module):
 def __init__(self,n,h,r):super().__init__();self.e=nn.Embedding(n,h);self.d=nn.Linear(h*2,r,bias=False);self.u=nn.Linear(r,h,bias=False);nn.init.zeros_(self.u.weight)
 def forward(self,x,i):return x+self.u(torch.tanh(self.d(torch.cat([x,self.e(i)],-1))))
class PointerEditor(nn.Module):
 def __init__(self,n,h,r,nodes):super().__init__();self.e=nn.Embedding(n,h);self.pointer=nn.Linear(h,nodes);self.node=nn.Embedding(nodes,h);self.d=nn.Linear(h*2,r,bias=False);self.u=nn.Linear(r,h,bias=False);nn.init.zeros_(self.u.weight)
 def forward(self,x,i,return_pointer=False):
  e=self.e(i);p=self.pointer(e);address=torch.softmax(p,-1)@self.node.weight;y=x+self.u(torch.tanh(self.d(torch.cat([x,e+address],-1))));return (y,p) if return_pointer else y
def metric(pred,label,fact):
 c=label!=fact;p=~c;ca=float((pred[c]==label[c]).float().mean());pa=float((pred[p]==label[p]).float().mean());return {"change_accuracy":ca,"preservation_accuracy":pa,"balanced_intervention_score":.5*(ca+pa),"change_decisions":int(c.sum()),"preservation_decisions":int(p.sum())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cell-id",required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--manifest-sha256",required=True);ap.add_argument("--amendment",type=Path,required=True);ap.add_argument("--amendment-sha256",required=True);a=ap.parse_args()
 if sha(a.manifest)!=a.manifest_sha256 or sha(a.amendment)!=a.amendment_sha256:raise ValueError("registry hash mismatch")
 m=json.loads(a.manifest.read_text());am=json.loads(a.amendment.read_text());xs=[x for x in m["cells"] if x["cell_id"]==a.cell_id]
 if len(xs)!=1 or am["base_manifest_sha256"]!=a.manifest_sha256:raise ValueError("unregistered cell")
 cell=xs[0];cache=Path(am["feature_cache"])
 if sha(cache)!=am["feature_cache_sha256"]:raise ValueError("cache mismatch")
 graph=next(x for x in m["graphs"] if x["graph_seed"]==cell["graph_seed"])
 for x in graph["files"].values():
  if sha(x["path"])!=x["sha256"]:raise ValueError("data mismatch")
 random.seed(cell["seed"]);torch.manual_seed(cell["seed"]);torch.cuda.manual_seed_all(cell["seed"]);dev=torch.device("cuda:0");c=torch.load(cache,map_location="cpu",weights_only=False);lookup={(x["graph_seed"],x["world_id"]):c["features"][i].float() for i,x in enumerate(c["records"])};nodes=m["operator_contract"]["nodes"];rows=json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text());worlds=json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text());wb={x["world_id"]:x["factual_state"] for x in worlds}
 def tensors(rs):return (torch.stack([lookup[(cell["graph_seed"],x["world_id"])] for x in rs]).to(dev),torch.tensor([nodes.index(x["intervention"]["target"])*2+int(x["intervention"]["value"]) for x in rs],device=dev),torch.tensor([[int(x["intervened_state"][n]) for n in nodes] for x in rs],device=dev),torch.tensor([[int(wb[x["world_id"]][n]) for n in nodes] for x in rs],device=dev))
 tr=tensors([x for x in rows if x["split"]=="train"]);va=tensors([x for x in rows if x["split"]=="validation"]);h=c["hidden_size"];arm=cell["arm"]
 op={"t0_no_instruction":Identity(),"t1_text":TextAdd(60,h),"t2a_naive":JointMLP(60,h,16),"t2b_encode_only":O2ConditionalLowRank(60,h,16),"t3a_conditioning":O3StateGated(60,h,16),"t3b_pointer":PointerEditor(60,h,16,30)}[arm].to(dev);head=StateHead(h,30).to(dev);params=list(op.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=m["training_contract"]["learning_rate"],weight_decay=1e-4);gen=torch.Generator().manual_seed(cell["seed"]);hist=[]
 for step in range(m["training_contract"]["steps"]):
  ix=torch.randint(0,len(tr[0]),(m["training_contract"]["batch_size"],),generator=gen).to(dev)
  if arm=="t3b_pointer":edited,pointer=op(tr[0][ix],tr[1][ix],True);loss=balanced_loss(head(edited),tr[2][ix],tr[2][ix]!=tr[3][ix])+.1*F.cross_entropy(pointer,tr[1][ix]//2)
  else:loss=balanced_loss(head(op(tr[0][ix],tr[1][ix])),tr[2][ix],tr[2][ix]!=tr[3][ix])
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
  if step in {0,1199} or (step+1)%200==0:hist.append({"step":step+1,"loss":float(loss.detach())})
 with torch.no_grad():pred=head(op(va[0],va[1])).argmax(-1);pointer_top1=float((op(va[0],va[1],True)[1].argmax(-1)==va[1]//2).float().mean()) if arm=="t3b_pointer" else None
 out=Path(cell["output"])
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);ck=out/"model.pt";torch.save({"operator":op.state_dict(),"head":head.state_dict(),"arm":arm,"test_evaluated":False},ck);s={"protocol":m["protocol"],"cell_id":cell["cell_id"],"arm":arm,"seed":cell["seed"],"graph_seed":cell["graph_seed"],"validation":metric(pred,va[2],va[3]),"pointer_top1":pointer_top1,"parameter_count":sum(p.numel() for p in params),"history":hist,"checkpoint_sha256":sha(ck),"manifest_sha256":a.manifest_sha256,"amendment_sha256":a.amendment_sha256,"feature_cache_sha256":am["feature_cache_sha256"],"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n");print(json.dumps(s))
if __name__=="__main__":main()
