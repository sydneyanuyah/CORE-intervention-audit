#!/usr/bin/env python3
"""Bind T3 cells to the 20 frozen A1 CLadder T3-b checkpoints."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=ROOT/"data/two_edit/cladder/artifact.json"; pairs=ROOT/"registry/t3_cladder_perturbations.json"; a1=json.loads((ROOT/"registry/a1_confirmatory_manifest.json").read_text()); cells=[]
 for seed in a1["seeds"]:
  checkpoint=ROOT/f"outputs/a1-confirmatory/cladder/t3b_pointer/seed-{seed}/best.pt"; summary=checkpoint.with_name("cell_summary.json"); contract=checkpoint.with_name("cell_contract.json")
  if not checkpoint.is_file() or not summary.is_file() or not contract.is_file() or json.loads(contract.read_text()).get("test_evaluated") is not False: raise ValueError(f"invalid frozen T3 source: {seed}")
  checkpoint_sha=sha(checkpoint); summary_sha=sha(summary); contract_sha=sha(contract)
  for view in ("clean","variable_rename","cladder_variants"):
   cells.append({"cell_id":f"t3:cladder:{view}:seed-{seed}","seed":seed,"view":view,"checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_sha,"source_summary":str(summary),"source_summary_sha256":summary_sha,"source_contract":str(contract),"source_contract_sha256":contract_sha,"world_size":4,"output":f"outputs/t3/cladder/{view}/seed-{seed}","test_evaluated":False})
 out={"protocol":"t3_cladder_frozen_robustness_v1","status":"preregistered","views":["clean","variable_rename","cladder_variants"],"records_per_view":144,"source_artifact":str(source),"source_artifact_sha256":sha(source),"perturbation_manifest":str(pairs),"perturbation_manifest_sha256":sha(pairs),"cells":cells,"test_evaluated":False}
 path=ROOT/"registry/t3_manifest.json"; path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"cells":len(cells),"sha256":sha(path)})); return 0
if __name__=="__main__": raise SystemExit(main())
