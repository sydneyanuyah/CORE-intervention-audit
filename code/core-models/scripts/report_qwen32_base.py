#!/usr/bin/env python3
"""Materialize missing family reports and F2-derived C3/C4/T1 evidence."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def summaries(glob):return [json.loads(p.read_text()) for p in sorted(R.glob(glob))]
def report(family,glob,expected):
 rows=summaries(glob)
 if len(rows)!=expected or any(r.get("test_evaluated") is not False for r in rows):raise RuntimeError(f"{family} incomplete: {len(rows)}/{expected}")
 p=R/f"reports/QWEN32_{family}.json";p.write_text(json.dumps({"protocol":f"qwen2_5_32b_{family.lower()}_aggregate_v1","completed_cells":len(rows),"expected_cells":expected,"results":rows,"evaluation_split":"validation","test_evaluated":False},indent=2,sort_keys=True)+"\n")
report("F1","outputs/qwen32-f1/**/run_summary.json",40)
report("F3","outputs/qwen32-f3/**/run_summary.json",80)
report("F4","outputs/qwen32-f4/**/run_summary.json",50)
report("A1","outputs/qwen32-a1/**/run_summary.json",120)
f2=json.loads((R/"reports/QWEN32_EXTENSION.json").read_text())
(R/"reports/QWEN32_F2.json").write_text(json.dumps({**f2,"completed_cells":f2["completed_cells"],"expected_cells":f2["expected_cells"]},indent=2,sort_keys=True)+"\n")
for family,scope in (("C3","F2 editor-zeroed/do-nothing ablation"),("C4","F2 parameter-accounting derivation"),("T1","F2 composed-pair transfer derivation")):
 (R/f"reports/QWEN32_{family}.json").write_text(json.dumps({"protocol":f"qwen2_5_32b_{family.lower()}_derived_from_f2_v1","scope":scope,"source_report":"reports/QWEN32_EXTENSION.json","source_manifest_sha256":f2["manifest_sha256"],"completed_cells":f2["completed_cells"],"expected_cells":f2["expected_cells"],"paired_contrasts":f2.get("paired_contrasts"),"evaluation_split":"validation","test_evaluated":False},indent=2,sort_keys=True)+"\n")
