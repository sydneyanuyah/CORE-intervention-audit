#!/usr/bin/env python3
"""Train one registered frozen-Phi F3 real/placebo cell."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from core_7b.operator_probe import O2ConditionalLowRank,O3StateGated,StateHead,balanced_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_rows(path):
 text=Path(path).read_text()
 try:return json.loads(text)
 except json.JSONDecodeError:return [json.loads(line) for line in text.splitlines() if line.strip()]
def metrics(pred,labels,facts):
 changed=labels!=facts;preserved=~changed
 change=float((pred[changed]==labels[changed]).float().mean())
 preserve=float((pred[preserved]==labels[preserved]).float().mean())
 return {"change_accuracy":change,"preservation_accuracy":preserve,"balanced_intervention_score":.5*(change+preserve),"change_decisions":int(changed.sum()),"preservation_decisions":int(preserved.sum())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cell-id",required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--manifest-sha256",required=True);ap.add_argument("--amendment",type=Path,required=True);ap.add_argument("--amendment-sha256",required=True);a=ap.parse_args()
 if sha(a.manifest)!=a.manifest_sha256 or sha(a.amendment)!=a.amendment_sha256:raise ValueError("registry hash mismatch")
 m=json.loads(a.manifest.read_text());am=json.loads(a.amendment.read_text());matches=[x for x in m["cells"] if x["cell_id"]==a.cell_id]
 if len(matches)!=1 or am["base_manifest_sha256"]!=a.manifest_sha256:raise ValueError("unregistered cell")
 cell=matches[0];cache=Path(am["feature_cache"])
 if sha(cache)!=am["feature_cache_sha256"]:raise ValueError("cache hash mismatch")
 graph=next(x for x in m["graphs"] if x["graph_seed"]==cell["graph_seed"])
 for x in graph["files"].values():
  if sha(x["path"])!=x["sha256"]:raise ValueError("data hash mismatch")
 random.seed(cell["seed"]);torch.manual_seed(cell["seed"]);torch.cuda.manual_seed_all(cell["seed"]);dev=torch.device("cuda:0")
 cache_data=torch.load(cache,map_location="cpu",weights_only=False);lookup={(x["graph_seed"],x["world_id"]):cache_data["features"][i].float() for i,x in enumerate(cache_data["records"])};nodes=m["operator_contract"]["nodes"]
 records=load_rows(graph["files"][f"records_{cell['condition']}.json"]["path"]);worlds=load_rows(graph["files"]["worlds.json"]["path"]);facts_by_world={x["world_id"]:x["factual_state"] for x in worlds}
 def tensors(rows):
  f=torch.stack([lookup[(cell["graph_seed"],x["world_id"])] for x in rows]).to(dev);ids=torch.tensor([nodes.index(x["intervention"]["target"])*2+int(x["intervention"]["value"]) for x in rows],device=dev);labels=torch.tensor([[int(x["intervened_state"][n]) for n in nodes] for x in rows],device=dev);facts=torch.tensor([[int(facts_by_world[x["world_id"]][n]) for n in nodes] for x in rows],device=dev);return f,ids,labels,facts
 train=tensors([x for x in records if x["split"]=="train"]);valid=tensors([x for x in records if x["split"]=="validation"])
 h=cache_data["hidden_size"];op=(O2ConditionalLowRank(60,h,16) if cell["method"]=="o2" else O3StateGated(60,h,16)).to(dev);head=StateHead(h,30).to(dev);params=list(op.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=m["training_contract"]["learning_rate"],weight_decay=1e-4);gen=torch.Generator().manual_seed(cell["seed"]);history=[]
 for step in range(m["training_contract"]["steps"]):
  ix=torch.randint(0,len(train[0]),(m["training_contract"]["batch_size"],),generator=gen).to(dev);logits=head(op(train[0][ix],train[1][ix]));loss=balanced_loss(logits,train[2][ix],train[2][ix]!=train[3][ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
  if step in {0,m["training_contract"]["steps"]-1} or (step+1)%200==0:history.append({"step":step+1,"loss":float(loss.detach())})
 with torch.no_grad():pred=head(op(valid[0],valid[1])).argmax(-1);dn=head(valid[0]).argmax(-1)
 out=Path(cell["output"])
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);ck=out/"operator.pt";torch.save({"operator":op.state_dict(),"head":head.state_dict(),"test_evaluated":False},ck)
 summary={"protocol":m["protocol"],"cell_id":cell["cell_id"],"condition":cell["condition"],"method":cell["method"],"seed":cell["seed"],"graph_seed":cell["graph_seed"],"clean":metrics(pred,valid[2],valid[3]),"do_nothing":metrics(dn,valid[2],valid[3]),"operator_parameter_count":sum(p.numel() for p in op.parameters()),"history":history,"checkpoint_sha256":sha(ck),"manifest_sha256":a.manifest_sha256,"amendment_sha256":a.amendment_sha256,"feature_cache_sha256":am["feature_cache_sha256"],"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary))
if __name__=="__main__":main()
