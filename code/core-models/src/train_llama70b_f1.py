#!/usr/bin/env python3
"""Train one registered Llama-3.3-70B F1 O2/O3 XOR cell."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from core_7b.operator_probe import O2ConditionalLowRank,O3StateGated,StateHead,balanced_loss,paired_metrics
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cell-id",required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--manifest-sha256",required=True);ap.add_argument("--amendment",type=Path,required=True);ap.add_argument("--amendment-sha256",required=True);a=ap.parse_args()
 if sha(a.manifest)!=a.manifest_sha256 or sha(a.amendment)!=a.amendment_sha256:raise ValueError("registry hash mismatch")
 m=json.loads(a.manifest.read_text());am=json.loads(a.amendment.read_text());cell=[x for x in m["cells"] if x["cell_id"]==a.cell_id]
 if len(cell)!=1 or am["base_manifest_sha256"]!=a.manifest_sha256:raise ValueError("unregistered cell");cell=cell[0]
 cell=cell[0] if isinstance(cell,list) else cell
 cache=Path(am["feature_cache"])
 if sha(cache)!=am["feature_cache_sha256"]:raise ValueError("cache hash mismatch")
 graph=[x for x in m["graphs"] if x["graph_seed"]==cell["graph_seed"]][0]
 for x in graph["files"].values():
  if sha(x["path"])!=x["sha256"]:raise ValueError("data hash mismatch")
 seed=cell["seed"];random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);dev=torch.device("cuda:0")
 c=torch.load(cache,map_location="cpu",weights_only=False);lookup={(x["graph_seed"],x["world_id"]):c["features"][i].float() for i,x in enumerate(c["records"])};nodes=m["operator_contract"]["nodes"]
 records=json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text());byid={x["record_id"]:x for x in records};train=[x for x in records if x["split"]=="train"]
 features=torch.stack([lookup[(cell["graph_seed"],x["world_id"])] for x in train]).to(dev);ids=torch.tensor([nodes.index(x["intervention"]["target"])*2+int(x["intervention"]["value"]) for x in train],device=dev)
 labels=torch.tensor([[int(x["intervened_state"][n]) for n in nodes] for x in train],device=dev)
 world_states={x["world_id"]:x["factual_state"] for x in json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text())};facts=torch.tensor([[int(world_states[x["world_id"]][n]) for n in nodes] for x in train],device=dev);changed=labels!=facts
 h=c["hidden_size"];op=(O2ConditionalLowRank(60,h,16) if cell["method"]=="o2" else O3StateGated(60,h,16)).to(dev);head=StateHead(h,30).to(dev);params=list(op.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=m["training_contract"]["learning_rate"],weight_decay=1e-4);gen=torch.Generator().manual_seed(seed)
 history=[]
 for step in range(m["training_contract"]["steps"]):
  ix=torch.randint(0,len(train),(m["training_contract"]["batch_size"],),generator=gen).to(dev);loss=balanced_loss(head(op(features[ix],ids[ix])),labels[ix],changed[ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
  if step in {0,m["training_contract"]["steps"]-1} or (step+1)%200==0:history.append({"step":step+1,"loss":float(loss.detach())})
 pairs=[json.loads(x) for x in Path(graph["files"]["pairs"]["path"]).read_text().splitlines() if x.strip() and json.loads(x)["split"]=="validation"]
 def predict(invert=False,none=False):
  out={};op.eval();head.eval()
  with torch.no_grad():
   for p in pairs:
    r1,r2=byid[p["first_record_id"]],byid[p["second_record_id"]];state=lookup[(cell["graph_seed"],r1["world_id"])].to(dev).unsqueeze(0)
    if not none:
     def iid(r):return nodes.index(r["intervention"]["target"])*2+(1-int(r["intervention"]["value"]) if invert else int(r["intervention"]["value"]))
     state=op(op(state,torch.tensor([iid(r1)],device=dev)),torch.tensor([iid(r2)],device=dev))
    out[p["pair_id"]]=head(state).argmax(-1)[0].tolist()
  return out
 clean,inv,dn=predict(),predict(True),predict(False,True);moved=sum(clean[k]!=inv[k] for k in clean);out=Path(cell["output"])
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);ck=out/"operator.pt";torch.save({"operator":op.state_dict(),"head":head.state_dict(),"test_evaluated":False},ck)
 s={"protocol":m["protocol"],"cell_id":cell["cell_id"],"method":cell["method"],"seed":seed,"graph_seed":cell["graph_seed"],"clean":paired_metrics(pairs,clean),"inverted_against_retained_gold":paired_metrics(pairs,inv),"do_nothing":paired_metrics(pairs,dn),"changed_pair_vectors":moved,"pair_count":len(pairs),"changed_pair_fraction":moved/len(pairs),"operator_parameter_count":sum(p.numel() for p in op.parameters()),"history":history,"checkpoint_sha256":sha(ck),"manifest_sha256":a.manifest_sha256,"amendment_sha256":a.amendment_sha256,"feature_cache_sha256":am["feature_cache_sha256"],"gpu_count":1,"evaluation_split":"validation","test_evaluated":False};(out/"run_summary.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n");print(json.dumps(s))
if __name__=="__main__":main()
