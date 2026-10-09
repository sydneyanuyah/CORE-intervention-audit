#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t4_manifest.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(xs):
 mean=statistics.fmean(xs); half=2.776445105*statistics.stdev(xs)/math.sqrt(len(xs)); return {"mean":mean,"student_t_95":[mean-half,mean+half]}
def main():
 m=json.loads(MP.read_text()); arms={}; cells=[]
 for arm in m["arms"]:
  vals=[]
  for seed in m["seeds"]:
   p=ROOT/f"outputs/t4/{arm}/seed-{seed}/run_summary.json"; d=json.loads(p.read_text())
   if d.get("test_evaluated") is not False or d.get("world_size")!=4 or d.get("arm")!=arm: raise ValueError("T4 provenance failed")
   vals.append(d["validation"]["balanced_direction_accuracy"]); cells.append({"arm":arm,"seed":seed,"sha256":sha(p)})
  arms[arm]=ci(vals)
 out={"protocol":m["protocol"],"manifest_sha256":sha(MP),"completed_cells":len(cells),"arms":arms,"source_cells":cells,"test_evaluated":False}; (ROOT/"reports/T4_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); (ROOT/"reports/T4_REPORT.md").write_text("# T4 rung transfer\n\n"+"\n".join(f"- {a}: {d['mean']:.6f} (95% [{d['student_t_95'][0]:.6f}, {d['student_t_95'][1]:.6f}])" for a,d in arms.items())+"\n"); return 0
if __name__=="__main__":raise SystemExit(main())
