#!/usr/bin/env python3
"""Close the 23-family Phi registry only after every report validates complete."""
import json
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];qpath=R/"registry/qwen32_23_queue.json";q=json.loads(qpath.read_text());reports={}
for e in q["experiments"]:
 p=R/f"reports/QWEN32_{e['id']}.json"
 if p.is_file():reports[e["id"]]=json.loads(p.read_text())
for e in q["experiments"]:
 if e["id"] in reports:
  r=reports[e["id"]
  ]
  if r.get("test_evaluated") is not False:raise ValueError(f"test lock failed {e['id']}")
  if "expected_cells" in r and r.get("completed_cells")!=r["expected_cells"]:raise ValueError(f"incomplete {e['id']}")
  e["state"]="complete";e["report"]=f"reports/QWEN32_{e['id']}.json";e["completed_cells"]=r.get("completed_cells",e.get("completed_cells"));e["expected_cells"]=r.get("expected_cells",e.get("expected_cells"))
if any(e["state"] not in {"complete","complete_derived"} for e in q["experiments"]):raise RuntimeError("not all 23 Phi families are complete")
q["current_engineering_order"]=[];q["state"]="complete";q["completed_at_utc"]=datetime.now(timezone.utc).isoformat();qpath.write_text(json.dumps(q,indent=2,sort_keys=True)+"\n")
with (R/"logs/RUNLOG.md").open("a") as f:f.write(f"\n- {q['completed_at_utc']}: Qwen2.5-32B expansion closed at 23/23 experiment families; all registered evaluation remained validation-only and `test_evaluated=false`.\n")
print(json.dumps({"complete_families":23,"test_evaluated":False}))
