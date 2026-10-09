#!/usr/bin/env python3
"""Aggregate complete model/task final-test cells without hiding missing cells."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASKS=("t2","t4","t5")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser(); p.add_argument("--models",nargs="+",required=True); a=p.parse_args()
for model in a.models:
 report={"protocol":f"{model}_expansion_final_one_shot_v1","model_key":model,"experiments":{},"excluded_sources":["com2"],"test_evaluated":True}
 for task in TASKS:
  mp=ROOT/f"registry/{model}_{task}_final_test_manifest.json"; m=json.loads(mp.read_text()); results=[]
  for cell in m["source_cells"]:
   path=ROOT/f"outputs/{model}-{task}-final-test/{cell['method']}/seed-{cell['seed']}/test_summary.json"
   if not path.is_file(): raise FileNotFoundError(path)
   row=json.loads(path.read_text())
   if row.get("manifest_sha256")!=sha(mp) or row.get("test_evaluated") is not True: raise ValueError(path)
   results.append(row)
  groups={}
  for method in sorted({r["method"] for r in results}):
   values=[r["test_accuracy"] for r in results if r["method"]==method]
   groups[method]={"cells":len(values),"mean_test_accuracy":statistics.mean(values),"standard_deviation":statistics.stdev(values) if len(values)>1 else 0.0}
  report["experiments"][task.upper()]={"manifest_sha256":sha(mp),"test_rows":m["test_rows"],"methods":groups,"results":results}
 report["experiments"]["A1"]={"status":"unscored","reason":"Expanded A1 was trained only on XOR; the sealed XOR bundle has labels and record IDs but no model input text.","test_evaluated":False}
 output=ROOT/f"reports/{model.upper()}_FINAL_ONE_SHOT.json"; output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(output)
