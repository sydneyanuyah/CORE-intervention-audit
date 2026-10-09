#!/usr/bin/env python3
"""Aggregate the completed held-out evaluation without reselection."""
from __future__ import annotations
import hashlib, json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/final-one-shot"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(values):
 values=list(map(float,values)); n=len(values); mean=statistics.fmean(values)
 critical={5:2.776445,20:2.093024}.get(n,1.96)
 half=critical*statistics.stdev(values)/math.sqrt(n) if n>1 else 0.0
 return {"n":n,"mean":mean,"student_t_95":[mean-half,mean+half],"values":values}
def read(paths):
 rows=[json.loads(p.read_text()) for p in sorted(paths)]
 if any(x.get("test_evaluated") is not True or x.get("world_size")!=4 for x in rows):raise RuntimeError("inadmissible final result")
 return rows

def main():
 status=json.loads((OUT/"RECOVERY_STATUS.json").read_text())
 if status!={"complete":120,"failures":[],"registered":120,"test_evaluated":True,"xor_unscored":20}:raise RuntimeError("final queue incomplete")
 a1={family:read((OUT/"a1"/family).glob("seed-*/test_summary.json")) for family in ("ccrgb","cladder","wiqa")}
 if any(len(v)!=20 for v in a1.values()):raise RuntimeError("A1 final cells incomplete")
 t2=read((OUT/"t2").glob("seed-*/test_summary.json"));
 t4={arm:read((OUT/"t4"/arm).glob("seed-*/test_summary.json")) for arm in ("changing_only","imagining_only","joint")}
 t5={method:read((OUT/"t5"/method).glob("seed-*/test_summary.json")) for method in ("baseline","o3")}
 if len(t2)!=5 or any(len(v)!=5 for v in t4.values()) or any(len(v)!=20 for v in t5.values()):raise RuntimeError("T/T5 final cells incomplete")
 a1_out={k:{"accuracy":stat(x["metrics"]["accuracy"] for x in v),"macro_f1":stat(x["metrics"]["macro_f1"] for x in v),"pointer_top1_accuracy":stat(x["metrics"]["pointer"]["pointer_top1_accuracy"] for x in v),"test_rows_per_seed":v[0]["test_rows"]} for k,v in a1.items()}
 t2_out={key:stat(x["metrics"][key] for x in t2) for key in ("balanced_direction_accuracy","change_accuracy","preservation_accuracy","accuracy")}
 t4_out={arm:{key:stat(x["metrics"][key] for x in rows) for key in ("balanced_direction_accuracy","change_accuracy","preservation_accuracy","accuracy")} for arm,rows in t4.items()}
 indexed={method:{x["seed"]:x for x in rows} for method,rows in t5.items()}
 if set(indexed["baseline"])!=set(indexed["o3"]):raise RuntimeError("T5 seed pairing failed")
 t5_out={method:{"accuracy":stat(x["metrics"]["accuracy"] for x in rows),"by_source":{source:stat(x["metrics"]["by_source"][source]["accuracy"] for x in rows) for source in ("counterbench","crass")}} for method,rows in t5.items()}
 t5_out["o3_minus_baseline_accuracy"]=stat(indexed["o3"][s]["metrics"]["accuracy"]-indexed["baseline"][s]["metrics"]["accuracy"] for s in sorted(indexed["o3"]))
 payload={"protocol":"core_final_one_shot_report_v1","manifest_sha256":json.loads((OUT/"ONE_SHOT_CONSUMED.json").read_text())["manifest_sha256"],"completed_cells":120,"failed_cells":0,"world_size_per_cell":4,"excluded_sources":["com2"],"a1":a1_out,"a1_xor":{"status":"unscored","cells":20,"reason":"sealed file contains composed truth/IDs but no model inputs and overlaps architecture-selection graphs"},"t2":t2_out,"t4":t4_out,"t5":t5_out,"test_evaluated":True}
 jp=ROOT/"reports/FINAL_ONE_SHOT.json";mp=ROOT/"reports/FINAL_ONE_SHOT.md";jp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
 lines=["# Final one-shot held-out evaluation","",f"Completed **120/120** admissible cells with zero runtime failures. Com2 was excluded; A1 XOR is unscored for the recorded data-contract reason.","","## One-line results","",*(f"- A1 {k}: accuracy {v['accuracy']['mean']:.6f} (95% [{v['accuracy']['student_t_95'][0]:.6f}, {v['accuracy']['student_t_95'][1]:.6f}]); pointer top-1 {v['pointer_top1_accuracy']['mean']:.6f}." for k,v in a1_out.items()),f"- T2: balanced direction accuracy {t2_out['balanced_direction_accuracy']['mean']:.6f} (change {t2_out['change_accuracy']['mean']:.6f}; preservation {t2_out['preservation_accuracy']['mean']:.6f}).",*(f"- T4 {arm}: balanced direction accuracy {v['balanced_direction_accuracy']['mean']:.6f}." for arm,v in t4_out.items()),f"- T5 baseline: accuracy {t5_out['baseline']['accuracy']['mean']:.6f}.",f"- T5 O3: accuracy {t5_out['o3']['accuracy']['mean']:.6f}.",f"- T5 O3 minus baseline: {t5_out['o3_minus_baseline_accuracy']['mean']:+.6f} (95% [{t5_out['o3_minus_baseline_accuracy']['student_t_95'][0]:+.6f}, {t5_out['o3_minus_baseline_accuracy']['student_t_95'][1]:+.6f}]).",""]
 mp.write_text("\n".join(lines));print(json.dumps({"json":str(jp),"json_sha256":sha(jp),"markdown":str(mp),"markdown_sha256":sha(mp)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
