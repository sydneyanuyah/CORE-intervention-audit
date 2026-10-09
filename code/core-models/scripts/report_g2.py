#!/usr/bin/env python3
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];m=json.loads((ROOT/"registry/g2_manifest.json").read_text()); vals={x:[] for x in ("o2","o3")}
for c in m["cells"]:vals[c["method"]].append(json.loads((ROOT/c["output"]/"measurement_summary.json").read_text())["validation"])
out={k:{"mean_hidden_agreement":statistics.mean(x["hidden_agreement_at_1e-5"] for x in v),"mean_normalized_hidden_gap":statistics.mean(x["mean_normalized_hidden_gap"] for x in v),"n":len(v)} for k,v in vals.items()};pairs=[vals["o3"][i]["hidden_agreement_at_1e-5"]-vals["o2"][i]["hidden_agreement_at_1e-5"] for i in range(20)];mean=statistics.mean(pairs);half=2.093*statistics.stdev(pairs)/math.sqrt(20) if statistics.stdev(pairs) else 0;out["paired_o3_minus_o2"]={"mean":mean,"student_t_95":[mean-half,mean+half]};(ROOT/"reports/G2_RESULTS.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");(ROOT/"reports/G2_RESULTS.md").write_text("# G2 results\n\n```json\n"+json.dumps(out,indent=2,sort_keys=True)+"\n```\n")
