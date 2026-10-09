#!/usr/bin/env python3
"""Aggregate Qwen2.5-32B C1 controls and C2 trivial floors."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def ci(xs):
 n=len(xs);mean=statistics.mean(xs);half=2.093*statistics.stdev(xs)/math.sqrt(n) if n>1 else 0.0;return [mean-half,mean+half]
def summary(path):return json.loads(Path(path).read_text())
def main():
 c1=[]
 for seed in range(601,621):
  for condition,path_condition in (("real","real"),("noncausal_placebo","placebo")):
   s=summary(ROOT/f"outputs/qwen32-f3/{path_condition}/o3/seed-{seed}/run_summary.json");c1.append({"seed":seed,"condition":condition,"score":s["clean"]["balanced_intervention_score"],"source_checkpoint_sha256":s["checkpoint_sha256"]})
  s=summary(ROOT/f"outputs/qwen32-c1/shuffled/o3/seed-{seed}/run_summary.json");c1.append({"seed":seed,"condition":"shuffled","score":s["clean"]["balanced_intervention_score"],"source_checkpoint_sha256":s["checkpoint_sha256"]})
 conditions={}
 for condition in ("real","noncausal_placebo","shuffled"):
  xs=[x["score"] for x in c1 if x["condition"]==condition];conditions[condition]={"mean":statistics.mean(xs),"student_t_95":ci(xs)}
 rows=[json.loads(x) for x in (ROOT/"data/experiments/c2/floor_suite.jsonl").read_text().splitlines() if x.strip()]
 floors={}
 for method in ("do_nothing","do_everything","random_init"):
  by_graph={}
  for r in rows:
   pred=r["floor_predictions"][method];gold=r["gold_state"];fact=r["factual_state"];changed=[k for k in gold if gold[k]!=fact[k]];preserved=[k for k in gold if gold[k]==fact[k]];a=sum(pred[k]==gold[k] for k in changed)/len(changed) if changed else 1.0;b=sum(pred[k]==gold[k] for k in preserved)/len(preserved) if preserved else 1.0;by_graph.setdefault(r["graph_group_id"],[]).append(.5*(a+b))
  xs=[statistics.mean(v) for v in by_graph.values()];floors[method]={"mean":statistics.mean(xs),"student_t_95":ci(xs),"graphs":len(xs),"cases":len(rows)}
 trained=[summary(ROOT/f"outputs/qwen32-f3/real/o3/seed-{s}/run_summary.json")["clean"]["balanced_intervention_score"] for s in range(601,621)];floors["trained_o3"]={"mean":statistics.mean(trained),"student_t_95":ci(trained),"graphs":20}
 common={"model":"Qwen/Qwen2.5-32B-Instruct","evaluation_split":"validation","test_evaluated":False};(ROOT/"reports/QWEN32_C1.json").write_text(json.dumps({"protocol":"qwen2_5_32b_c1_aggregate_v1",**common,"conditions":conditions,"cells":c1},indent=2,sort_keys=True)+"\n");(ROOT/"reports/QWEN32_C2.json").write_text(json.dumps({"protocol":"qwen2_5_32b_c2_floor_calibration_v1",**common,"methods":floors},indent=2,sort_keys=True)+"\n");print(json.dumps({"c1_cells":60,"c2_cases":len(rows),"test_evaluated":False}))
if __name__=="__main__":main()
