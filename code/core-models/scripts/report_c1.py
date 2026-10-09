#!/usr/bin/env python3
"""Aggregate the 60 provenance-valid C1 cells at graph level."""
from __future__ import annotations
import hashlib,json,math,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def tint(v):
 m=statistics.fmean(v); h=2.093024054*statistics.stdev(v)/math.sqrt(len(v)); return [m-h,m+h]
def boot(v):
 r=random.Random(401); x=sorted(statistics.fmean(r.choice(v) for _ in v) for _ in range(10000)); return [x[249],x[9749]]
def main():
 mp=ROOT/"registry/c1_manifest.json"; cp=ROOT/"registry/c1_cells.json"; m=load(mp); c=load(cp)
 if c.get("manifest_sha256")!=sha(mp) or len(c.get("cells",[]))!=60: raise RuntimeError("C1 registry invalid")
 values={k:[] for k in m["conditions"]}; cells=[]
 for cell in c["cells"]:
  p=ROOT/f"outputs/c1/{cell['condition']}/seed-{cell['seed']}/run_summary.json"; d=load(p)
  if d.get("protocol")!=m["protocol"] or d.get("cell_id")!=cell["cell_id"] or d.get("test_evaluated") is not False: raise RuntimeError(f"invalid C1 source {p}")
  score=float(d["composed"]["two_edit_balanced"]); values[cell["condition"]].append(score); cells.append({"cell_id":cell["cell_id"],"score":score,"summary_sha256":sha(p)})
 real,placebo=values["real"],values["noncausal_placebo"]; delta=[a-b for a,b in zip(real,placebo)]
 result={"protocol":m["protocol"],"manifest_sha256":sha(mp),"cells_sha256":sha(cp),"cell_count":60,"condition_means":{k:statistics.fmean(v) for k,v in values.items()},"primary_real_minus_placebo":{"mean":statistics.fmean(delta),"student_t_95":tint(delta),"paired_graph_bootstrap_95":boot(delta),"paired_graph_deltas":delta},"cells":cells,"test_evaluated":False}
 (ROOT/"reports/C1_EVIDENCE.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 p=result["primary_real_minus_placebo"]; lines=["# C1 Closing controls","","Validation-only fixed-O3 results over 20 paired graph units; no held-out test access.","",f"Real mean: {result['condition_means']['real']:.6f}",f"Noncausal placebo mean: {result['condition_means']['noncausal_placebo']:.6f}",f"Shuffled mean: {result['condition_means']['shuffled']:.6f}","",f"Primary real − placebo: {p['mean']:+.6f}; 95% t [{p['student_t_95'][0]:+.6f}, {p['student_t_95'][1]:+.6f}]; paired graph bootstrap [{p['paired_graph_bootstrap_95'][0]:+.6f}, {p['paired_graph_bootstrap_95'][1]:+.6f}]."]
 (ROOT/"reports/C1_REPORT.md").write_text("\n".join(lines)+"\n"); return 0
if __name__=="__main__": raise SystemExit(main())
