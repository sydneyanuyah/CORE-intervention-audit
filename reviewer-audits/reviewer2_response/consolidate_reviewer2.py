#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]);raw=base/"raw";reports=[];seed_rows=[]
for p in sorted(raw.glob("*_a3_address_inversion.json")):
 d=json.loads(p.read_text());reports.append({"backbone":d["backbone"],"prefix":d["prefix"],"n_seeds":d["n_seeds"],"aggregate":d["aggregate"],"source_manifest_sha256":d["source_manifest_sha256"],"source_amendment_sha256":d["source_amendment_sha256"]})
 for r in d["per_seed"]:
  row={"backbone":d["backbone"],"prefix":d["prefix"],"seed":r["seed"],"graph_seed":r["graph_seed"]}
  for arm in ("correct","wrong_target","inverted_value"):
   for metric,value in r["arms"][arm].items():row[f"{arm}_{metric}"]=value
  for k,v in r.items():
   if k not in {"seed","graph_seed","arms"}:row[k]=v
  seed_rows.append(row)
out={"protocol":"reviewer2_a3_address_inversion_consolidated_v1","split":"validation","test_evaluated":False,"backbones":reports,"per_seed":seed_rows}
(base/"CORE_REVIEWER2_A3_ADDRESS_INVERSION.json").write_text(json.dumps(out,indent=2)+"\n")
with (base/"CORE_REVIEWER2_A3_ADDRESS_INVERSION_PER_SEED.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(seed_rows[0]));w.writeheader();w.writerows(seed_rows)
(base/"CORE_REVIEWER2_ORACLE_RESIDUAL_CALIBRATION.json").write_bytes((raw/"oracle_residual_calibration.json").read_bytes())
manifest=[]
for p in sorted(x for x in base.rglob("*") if x.is_file() and x.name!="SHA256SUMS.json"):
 manifest.append({"path":str(p.relative_to(base)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(base/"SHA256SUMS.json").write_text(json.dumps({"files":manifest},indent=2)+"\n")
print(json.dumps({"backbones":len(reports),"per_seed_rows":len(seed_rows)},indent=2))
