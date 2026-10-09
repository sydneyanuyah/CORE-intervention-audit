#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];MP=ROOT/"registry/g3_repaired_manifest.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads(MP.read_text());rows=[]
 for layer in m["candidate_layers"]:
  p=ROOT/f"outputs/g3-repaired/layer-{layer}/run_summary.json";d=json.loads(p.read_text())
  if d.get("test_evaluated") is not False or d.get("world_size")!=4 or d.get("candidate_layer")!=layer:raise ValueError("G3 provenance failed")
  rows.append({"layer":layer,"balanced_direction_accuracy":d["validation"]["balanced_direction_accuracy"],"summary_sha256":sha(p)})
 selected=sorted(rows,key=lambda x:(-x["balanced_direction_accuracy"],x["layer"]))[0];out={"protocol":m["protocol"],"manifest_sha256":sha(MP),"completed_cells":11,"ranked_layers":sorted(rows,key=lambda x:-x["balanced_direction_accuracy"]),"selected_layer":selected["layer"],"selection_metric":m["selection"]["metric"],"test_evaluated":False};(ROOT/"reports/G3_REPAIRED_SELECTION.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");(ROOT/"reports/G3_REPAIRED_SELECTION.md").write_text(f"# G3 repaired layer selection\n\nSelected layer **{selected['layer']}** at validation balanced direction accuracy **{selected['balanced_direction_accuracy']:.6f}**.\n");return 0
if __name__=="__main__":raise SystemExit(main())
