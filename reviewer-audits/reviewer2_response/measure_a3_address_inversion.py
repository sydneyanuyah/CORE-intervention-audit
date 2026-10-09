#!/usr/bin/env python3
"""Frozen-checkpoint A3 wrong-target and inverted-value sensitivity."""
import argparse,hashlib,importlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import torch
from core_7b.operator_probe import StateHead
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def t95(xs):
 m=statistics.fmean(xs);q={20:2.093024054408263}[len(xs)];e=q*statistics.stdev(xs)/math.sqrt(len(xs));return [m-e,m+e]
def metric(pred,label,fact,target):
 c=label!=fact;p=~c;ca=float((pred[c]==label[c]).float().mean());pa=float((pred[p]==label[p]).float().mean());ts=float((pred[torch.arange(len(pred),device=pred.device),target]==label[torch.arange(len(pred),device=pred.device),target]).float().mean());return {"change_accuracy":ca,"preservation_accuracy":pa,"balanced_accuracy":.5*(ca+pa),"target_success":ts}
def main():
 q=argparse.ArgumentParser();q.add_argument("--prefix",required=True);q.add_argument("--output",type=Path,required=True);a=q.parse_args();prefix=a.prefix
 M=ROOT/f"registry/{prefix}_a1_manifest.json";A=ROOT/f"registry/{prefix}_a1_cache_amendment.json";m=json.loads(M.read_text());am=json.loads(A.read_text());module=importlib.import_module(f"train_{prefix}_a1");PointerEditor=module.PointerEditor
 if am["base_manifest_sha256"]!=sha(M) or sha(am["feature_cache"])!=am["feature_cache_sha256"]:raise ValueError("binding mismatch")
 dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");cache=torch.load(am["feature_cache"],map_location="cpu",weights_only=False);lookup={(x["graph_seed"],x["world_id"]):cache["features"][i].float() for i,x in enumerate(cache["records"])};nodes=m["operator_contract"]["nodes"];cells={(x["seed"],x["arm"]):x for x in m["cells"]};rows=[]
 for seed in m["seeds"]:
  cell=cells[(seed,"t3b_pointer")];graph=next(x for x in m["graphs"] if x["graph_seed"]==cell["graph_seed"]);records=[x for x in json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text()) if x["split"]=="validation"];worlds={x["world_id"]:x["factual_state"] for x in json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text())};features=torch.stack([lookup[(graph["graph_seed"],x["world_id"])] for x in records]).to(dev);ids=torch.tensor([nodes.index(x["intervention"]["target"])*2+int(x["intervention"]["value"]) for x in records],device=dev);target=ids//2;labels=torch.tensor([[int(x["intervened_state"][n]) for n in nodes] for x in records],device=dev);facts=torch.tensor([[int(worlds[x["world_id"]][n]) for n in nodes] for x in records],device=dev)
  ck=ROOT/cell["output"]/"model.pt";sp=ROOT/cell["output"]/"run_summary.json";summary=json.loads(sp.read_text());
  if sha(ck)!=summary["checkpoint_sha256"] or summary["test_evaluated"] is not False:raise ValueError("checkpoint provenance failure")
  op=PointerEditor(60,cache["hidden_size"],16,30).to(dev);head=StateHead(cache["hidden_size"],30).to(dev);state=torch.load(ck,map_location=dev,weights_only=False);op.load_state_dict(state["operator"]);head.load_state_dict(state["head"]);op.eval();head.eval();wrong=((target+7)%30)*2+(ids%2);inverted=target*2+(1-ids%2)
  with torch.inference_mode():
   pc,ac=op(features,ids,True);pw,aw=op(features,wrong,True);pi,ai=op(features,inverted,True);pc=head(pc).argmax(-1);pw=head(pw).argmax(-1);pi=head(pi).argmax(-1)
  arms={"correct":metric(pc,labels,facts,target),"wrong_target":metric(pw,labels,facts,target),"inverted_value":metric(pi,labels,facts,target)}
  rows.append({"seed":seed,"graph_seed":graph["graph_seed"],"arms":arms,"wrong_minus_correct_balanced":arms["wrong_target"]["balanced_accuracy"]-arms["correct"]["balanced_accuracy"],"wrong_minus_correct_target_success":arms["wrong_target"]["target_success"]-arms["correct"]["target_success"],"inverted_minus_correct_balanced":arms["inverted_value"]["balanced_accuracy"]-arms["correct"]["balanced_accuracy"],"inverted_minus_correct_target_success":arms["inverted_value"]["target_success"]-arms["correct"]["target_success"],"wrong_decision_vector_changed_fraction":float((pw!=pc).any(-1).float().mean()),"inverted_decision_vector_changed_fraction":float((pi!=pc).any(-1).float().mean()),"correct_pointer_top1":float((ac.argmax(-1)==target).float().mean()),"wrong_pointer_top1_against_true":float((aw.argmax(-1)==target).float().mean()),"inverted_pointer_top1_against_true":float((ai.argmax(-1)==target).float().mean()),"checkpoint_sha256":summary["checkpoint_sha256"],"metric_source_path":str(sp.relative_to(ROOT)),"metric_source_sha256":sha(sp)})
 fields=["wrong_minus_correct_balanced","wrong_minus_correct_target_success","inverted_minus_correct_balanced","inverted_minus_correct_target_success","wrong_decision_vector_changed_fraction","inverted_decision_vector_changed_fraction"]
 agg={f:{"mean":statistics.fmean(x[f] for x in rows),"student_t_95":t95([x[f] for x in rows])} for f in fields};out={"protocol":"reviewer2_a3_address_inversion_existing_checkpoints_v1","backbone":m["model"],"prefix":prefix,"split":"validation","test_evaluated":False,"n_seeds":len(rows),"wrong_target_rule":"(true target index + 7) mod 30, commanded value unchanged","inverted_value_rule":"true target unchanged, commanded binary value inverted","per_seed":rows,"aggregate":agg,"source_manifest_sha256":sha(M),"source_amendment_sha256":sha(A)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
if __name__=="__main__":main()
