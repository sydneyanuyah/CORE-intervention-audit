#!/usr/bin/env python3
"""Inference-free oracle calibration on executable validation SCMs."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from core_bert.xor_two_edit import xor_scm_state,generate_xor_two_edit_manifest_from_directory
from core_bert.ccrgb_two_edit_adapter import solve_ccrgb_world
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hamming(a,b,nodes):return sum(a[x]!=b[x] for x in nodes)/len(nodes)
def summarize(vals):return {"count":len(vals),"mean":sum(vals)/len(vals),"max":max(vals)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 f2=json.loads((ROOT/"registry/f2_manifest.json").read_text()); out={"protocol":"reviewer2_oracle_residual_calibration_v1","split":"validation","test_evaluated":False,"families":{}}
 # XOR: all registered F2 graph seeds, validation worlds and scheduled two-edit pairs.
 xr={k:[] for k in ["identity_residual","identity_correct","idempotence_residual","idempotence_correct","commutation_residual","commutation_correct"]}; xprov=[]
 for seed in f2["xor_graph_seeds"]:
  d=Path(f2["paths"]["xor_artifact_template"].format(graph_seed=seed)); graph=json.loads((d/"graph.json").read_text()); worlds=json.loads((d/"worlds.json").read_text()); records=json.loads((d/"records_real.json").read_text()); pairs=generate_xor_two_edit_manifest_from_directory(d); nodes=graph["nodes"]; wb={w["world_id"]:w for w in worlds}; rb={r["record_id"]:r for r in records}
  for w in worlds:
   if w["split"]!="validation":continue
   oracle=xor_scm_state(graph,w["root_values"],{});xr["identity_residual"].append(hamming(oracle,w["factual_state"],nodes));xr["identity_correct"].append(float(oracle==w["factual_state"]))
  for r in records:
   if r["split"]!="validation":continue
   w=wb[r["world_id"]];i=r["intervention"];once=xor_scm_state(graph,w["root_values"],{i["target"]:i["value"]});twice=xor_scm_state(graph,w["root_values"],{i["target"]:i["value"]});xr["idempotence_residual"].append(hamming(once,twice,nodes));xr["idempotence_correct"].append(float(once==r["intervened_state"]))
  for p in pairs:
   if p["split"]!="validation":continue
   r1,r2=rb[p["first_record_id"]],rb[p["second_record_id"]];w=wb[r1["world_id"]];i1,i2=r1["intervention"],r2["intervention"]
   if i1["target"]==i2["target"]:continue
   lr=xor_scm_state(graph,w["root_values"],{i1["target"]:i1["value"],i2["target"]:i2["value"]});rl=xor_scm_state(graph,w["root_values"],{i2["target"]:i2["value"],i1["target"]:i1["value"]});gold={n:v for n,v in zip(nodes,p["gold_outputs"])};xr["commutation_residual"].append(hamming(lr,rl,nodes));xr["commutation_correct"].append(float(lr==gold and rl==gold))
  xprov.append({"graph_seed":seed,"graph_sha256":sha(d/"graph.json"),"worlds_sha256":sha(d/"worlds.json"),"records_sha256":sha(d/"records_real.json")})
 out["families"]["xor"]={k:summarize(v) for k,v in xr.items()};out["families"]["xor"]["provenance"]=xprov
 # CCR.GB: current executable artifact plus registered pair manifest.
 apath=ROOT/f2["paths"]["ccrgb_artifact"];mpath=ROOT/f2["paths"]["ccrgb_manifest"];art=json.loads(apath.read_text()); pairs=[json.loads(x) for x in mpath.read_text().splitlines() if x.strip()];worlds={w["context_id"]:w for w in art["worlds"]};comps={c["id"]:c for c in art["components"]};cr={k:[] for k in xr}
 for w in worlds.values():
  if w["split"]!="validation":continue
  oracle=solve_ccrgb_world(w,{});nodes=w["dag_nodes"];cr["identity_residual"].append(hamming(oracle,w["factual_state"],nodes));cr["identity_correct"].append(float(oracle==w["factual_state"]))
 for c in comps.values():
  if c["two_edit_metadata"]["split"]!="validation":continue
  w=worlds[c["two_edit_metadata"]["context_id"]];nodes=w["dag_nodes"];i=c["intervention"];once=solve_ccrgb_world(w,{i["target"]:i["value"]});twice=solve_ccrgb_world(w,{i["target"]:i["value"]});cr["idempotence_residual"].append(hamming(once,twice,nodes));cr["idempotence_correct"].append(float(once==c["intervened"]["state"]))
 for p in pairs:
  if p["split"]!="validation":continue
  c1,c2=comps[p["first_record_id"]],comps[p["second_record_id"]];w=worlds[c1["two_edit_metadata"]["context_id"]];nodes=w["dag_nodes"];i1,i2=c1["intervention"],c2["intervention"]
  if i1["target"]==i2["target"]:continue
  lr=solve_ccrgb_world(w,{i1["target"]:i1["value"],i2["target"]:i2["value"]});rl=solve_ccrgb_world(w,{i2["target"]:i2["value"],i1["target"]:i1["value"]});gold={n:v for n,v in zip(nodes,p["gold_outputs"])};cr["commutation_residual"].append(hamming(lr,rl,nodes));cr["commutation_correct"].append(float(lr==gold and rl==gold))
 out["families"]["ccrgb"]={k:summarize(v) for k,v in cr.items()};out["families"]["ccrgb"]["provenance"]={"artifact_path":str(apath.relative_to(ROOT)),"artifact_sha256":sha(apath),"manifest_path":str(mpath.relative_to(ROOT)),"manifest_sha256":sha(mpath)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({f:{k:v["mean"] for k,v in x.items() if isinstance(v,dict) and "mean" in v} for f,x in out["families"].items()},indent=2))
if __name__=="__main__":main()
