#!/usr/bin/env python3
"""Measure A2 controls and A3 address sensitivity from frozen Phi A1 cells."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import torch
from core_7b.operator_probe import O2ConditionalLowRank,StateHead
from train_qwen32_a1 import Identity,TextAdd,PointerEditor
M=ROOT/"registry/qwen32_a1_manifest.json";A=ROOT/"registry/qwen32_a1_cache_amendment.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(pred,label,fact):
 c=label!=fact;p=~c;ca=float((pred[c]==label[c]).float().mean());pa=float((pred[p]==label[p]).float().mean());return {"change_accuracy":ca,"preservation_accuracy":pa,"balanced_intervention_score":.5*(ca+pa),"change_decisions":int(c.sum()),"preservation_decisions":int(p.sum())}
def load_arm(cell,arm,h,dev):
 ck=Path(cell["output"])/"model.pt";summary=json.loads((Path(cell["output"])/"run_summary.json").read_text())
 if sha(ck)!=summary["checkpoint_sha256"] or summary["test_evaluated"] is not False:raise ValueError("invalid source checkpoint")
 op={"t0_no_instruction":Identity(),"t1_text":TextAdd(60,h),"t2b_encode_only":O2ConditionalLowRank(60,h,16),"t3b_pointer":PointerEditor(60,h,16,30)}[arm].to(dev);head=StateHead(h,30).to(dev);state=torch.load(ck,map_location=dev,weights_only=False);op.load_state_dict(state["operator"]);head.load_state_dict(state["head"]);op.eval();head.eval();return op,head,summary
def main():
 m=json.loads(M.read_text());am=json.loads(A.read_text());ms=sha(M);ams=sha(A)
 if am["base_manifest_sha256"]!=ms or sha(am["feature_cache"])!=am["feature_cache_sha256"]:raise ValueError("frozen binding mismatch")
 if sum((ROOT/x["output"]/"run_summary.json").is_file() for x in m["cells"])!=120:raise RuntimeError("A1 incomplete")
 dev=torch.device("cuda:0");cache=torch.load(am["feature_cache"],map_location="cpu",weights_only=False);lookup={(x["graph_seed"],x["world_id"]):cache["features"][i].float() for i,x in enumerate(cache["records"])};nodes=m["operator_contract"]["nodes"];cells={(x["seed"],x["arm"]):x for x in m["cells"]};a2=[];a3=[]
 for seed in m["seeds"]:
  graph=next(x for x in m["graphs"] if x["graph_seed"]==cells[(seed,"t2b_encode_only")]["graph_seed"]);rows=[x for x in json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text()) if x["split"]=="validation"];worlds=json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text());wb={x["world_id"]:x["factual_state"] for x in worlds};features=torch.stack([lookup[(graph["graph_seed"],x["world_id"])] for x in rows]).to(dev);ids=torch.tensor([nodes.index(x["intervention"]["target"])*2+int(x["intervention"]["value"]) for x in rows],device=dev);labels=torch.tensor([[int(x["intervened_state"][n]) for n in nodes] for x in rows],device=dev);facts=torch.tensor([[int(wb[x["world_id"]][n]) for n in nodes] for x in rows],device=dev)
  with torch.no_grad():
   t2,h2,s2=load_arm(cells[(seed,"t2b_encode_only")],"t2b_encode_only",cache["hidden_size"],dev);active=h2(t2(features,ids)).argmax(-1);zero=h2(features).argmax(-1)
   t0,h0,s0=load_arm(cells[(seed,"t0_no_instruction")],"t0_no_instruction",cache["hidden_size"],dev);no_instruction=h0(t0(features,ids)).argmax(-1)
   t1,h1,s1=load_arm(cells[(seed,"t1_text")],"t1_text",cache["hidden_size"],dev);prompting=h1(t1(features,ids)).argmax(-1)
   ptr,hp,sp=load_arm(cells[(seed,"t3b_pointer")],"t3b_pointer",cache["hidden_size"],dev);correct,p_correct=ptr(features,ids,True);random_ids=((ids//2+7)%30)*2+(ids%2);random,p_random=ptr(features,random_ids,True);correct_pred=hp(correct).argmax(-1);random_pred=hp(random).argmax(-1)
  a2.append({"seed":seed,"graph_seed":graph["graph_seed"],"active":metric(active,labels,facts),"editor_zeroed":metric(zero,labels,facts),"no_instruction":metric(no_instruction,labels,facts),"prompting":metric(prompting,labels,facts),"source_checkpoints":{"t2b":s2["checkpoint_sha256"],"t0":s0["checkpoint_sha256"],"t1":s1["checkpoint_sha256"]}})
  a3.append({"seed":seed,"graph_seed":graph["graph_seed"],"correct_address":metric(correct_pred,labels,facts),"random_address":metric(random_pred,labels,facts),"correct_pointer_top1":float((p_correct.argmax(-1)==ids//2).float().mean()),"random_pointer_top1_against_true":float((p_random.argmax(-1)==ids//2).float().mean()),"decision_vector_changed_fraction":float((correct_pred!=random_pred).any(-1).float().mean()),"source_checkpoint":sp["checkpoint_sha256"]})
 out=ROOT/"reports";out.mkdir(exist_ok=True);common={"source_manifest_sha256":ms,"source_amendment_sha256":ams,"evaluation_split":"validation","test_evaluated":False};(out/"QWEN32_A2.json").write_text(json.dumps({"protocol":"qwen2_5_32b_a2_controls_v1",**common,"cells":a2},indent=2,sort_keys=True)+"\n");(out/"QWEN32_A3.json").write_text(json.dumps({"protocol":"qwen2_5_32b_a3_address_sensitivity_v1",**common,"cells":a3},indent=2,sort_keys=True)+"\n");print(json.dumps({"a2_cells":len(a2),"a3_cells":len(a3),"test_evaluated":False}))
if __name__=="__main__":main()
