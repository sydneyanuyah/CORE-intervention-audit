#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(xs):
 mean=statistics.fmean(xs);half=2.776445105*statistics.stdev(xs)/math.sqrt(len(xs));return {"mean":mean,"student_t_95":[mean-half,mean+half]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--task",choices=("t5","g1","g4"),required=True);a=p.parse_args();mp=ROOT/f"registry/{a.task}_manifest.json";m=json.loads(mp.read_text());values={method:[] for method in m["methods"]};cells=[]
 for seed in m["seeds"]:
  for method in m["methods"]:
   path=ROOT/f"outputs/{a.task}/{method}/seed-{seed}/run_summary.json";d=json.loads(path.read_text())
   if d.get("test_evaluated") is not False or d.get("world_size")!=4:raise ValueError("candidate-transfer provenance failed")
   values[method].append(d["validation"]["accuracy"]);cells.append({"method":method,"seed":seed,"summary_sha256":sha(path)})
 delta=[x-y for x,y in zip(values["o3"],values["baseline"])];out={"protocol":m["protocol"],"manifest_sha256":sha(mp),"methods":{k:stats(v) for k,v in values.items()},"o3_minus_baseline":stats(delta),"source_cells":cells,"test_evaluated":False};(ROOT/f"reports/{a.task.upper()}_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");(ROOT/f"reports/{a.task.upper()}_REPORT.md").write_text(f"# {a.task.upper()} candidate transfer\n\nO3 minus baseline: {out['o3_minus_baseline']['mean']:+.6f} (95% [{out['o3_minus_baseline']['student_t_95'][0]:+.6f}, {out['o3_minus_baseline']['student_t_95'][1]:+.6f}]).\n");return 0
if __name__=="__main__":raise SystemExit(main())
