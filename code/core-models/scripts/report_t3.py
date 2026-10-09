#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t3_manifest.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(xs):
 mean=statistics.fmean(xs); half=2.093024054*statistics.stdev(xs)/math.sqrt(len(xs)); return {"mean":mean,"student_t_95":[mean-half,mean+half],"n":len(xs)}
def main():
 m=json.loads(MP.read_text()); by_seed=defaultdict(dict); sources=[]
 for cell in m["cells"]:
  p=ROOT/cell["output"]/"measurement_summary.json"; d=json.loads(p.read_text())
  if d.get("test_evaluated") is not False or d.get("checkpoint_sha256")!=cell["checkpoint_sha256"] or d.get("distributed",{}).get("world_size")!=4: raise ValueError(f"T3 provenance failed: {cell['cell_id']}")
  by_seed[cell["seed"]][cell["view"]]=float(d["validation"]["accuracy"]); sources.append({"cell_id":cell["cell_id"],"summary_sha256":sha(p),"accuracy":d["validation"]["accuracy"]})
 results={}
 for view in ("variable_rename","cladder_variants"):
  deltas=[views[view]-views["clean"] for views in by_seed.values()]; results[f"{view}_minus_clean"]=ci(deltas)
 out={"protocol":m["protocol"],"manifest_sha256":sha(MP),"completed_cells":len(sources),"accuracy_deltas":results,"source_cells":sources,"test_evaluated":False}; (ROOT/"reports/T3_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 lines=["# T3 CLadder robustness","",f"Completed {len(sources)} frozen-checkpoint four-GPU validation measurements.","","| Contrast | Mean accuracy delta | 95% CI |","|---|---:|---:|"]
 for name,d in results.items(): lines.append(f"| {name} | {d['mean']:+.6f} | [{d['student_t_95'][0]:+.6f}, {d['student_t_95'][1]:+.6f}] |")
 (ROOT/"reports/T3_REPORT.md").write_text("\n".join(lines)+"\n"); print(json.dumps(results,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
