#!/usr/bin/env python3
"""Resume untouched final cells after the pre-inference XOR adapter failure."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cmd(c,m):
 t=str(ROOT/".venv/bin/torchrun"); common=[t,"--standalone","--nproc_per_node=4"]; cp=str(ROOT/c["checkpoint"]); out=str(ROOT/c["output"])
 if c["task"]=="a1":
  data=ROOT/f"experiments/Experiment-5-A1/data/test/{c['family']}.jsonl"
  return common+[str(ROOT/"src/evaluate_a1_final.py"),"--checkpoint",cp,"--checkpoint-sha256",c["checkpoint_sha256"],"--test-data",str(data),"--source",c["family"],"--output",out,"--authorization","FINAL_ONE_SHOT"]
 if c["task"] in {"t2","t4"}:
  task=c["task"]; a=common+[str(ROOT/"src/evaluate_t2_t4_final.py"),"--task",task,"--checkpoint",cp,"--checkpoint-sha256",c["checkpoint_sha256"],"--test-data",str(ROOT/m["test_sets"][task]["path"]),"--manifest",str(ROOT/m["source_manifests"][task]["path"]),"--manifest-sha256",m["source_manifests"][task]["sha256"],"--output",out,"--authorization","FINAL_ONE_SHOT"]
  return a+(["--arm",c["arm"]] if task=="t4" else [])
 return common+[str(ROOT/"src/evaluate_candidate_final.py"),"--method",c["method"],"--seed",str(c["seed"]),"--checkpoint",cp,"--checkpoint-sha256",c["checkpoint_sha256"],"--test-data",str(ROOT/m["test_sets"]["t5"]["path"]),"--manifest",str(ROOT/m["source_manifests"]["t5"]["path"]),"--manifest-sha256",m["source_manifests"]["t5"]["sha256"],"--output",out,"--authorization","FINAL_ONE_SHOT"]

def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256: raise ValueError("manifest identity mismatch")
 m=json.loads(a.manifest.read_text()); root=ROOT/"outputs/final-one-shot"; consumed=json.loads((root/"ONE_SHOT_CONSUMED.json").read_text())
 if consumed.get("manifest_sha256")!=a.manifest_sha256: raise ValueError("consumed lock mismatch")
 eligible=[c for c in m["cells"] if not(c["task"]=="a1" and c.get("family")=="xor")]
 complete=[c for c in eligible if (ROOT/c["output"]/"test_summary.json").is_file()]
 cells=[c for c in eligible if not (ROOT/c["output"]/"test_summary.json").is_file()]
 if len(eligible)!=120 or any((ROOT/c["output"]).exists() for c in cells): raise RuntimeError("pending output path is not clean")
 provenance={"protocol":"final_one_shot_continuation_v2","original_manifest_sha256":a.manifest_sha256,"excluded_failed_cells":20,"reason":"resume only untouched cells after correcting physical-test split metadata validation","runner_sha256":{x:sha(ROOT/x) for x in ["src/evaluate_a1_final.py","src/evaluate_t2_t4_final.py","src/evaluate_candidate_final.py"]},"completed_before":len(complete),"pending_cells":len(cells)}
 (root/"CONTINUATION_PROVENANCE.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
 pending=iter(cells); active={}; failures=[]; logs=ROOT/"logs/final-one-shot"
 while True:
  for group in GROUPS:
   if group in active: continue
   try:c=next(pending)
   except StopIteration:break
   env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=group;h=(logs/f"recovery-{c['cell_id'].replace(':','-')}.log").open("a");proc=subprocess.Popen(cmd(c,m),cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);active[group]=(proc,h,c)
  if not active:break
  time.sleep(2)
  for group,(proc,h,c) in list(active.items()):
   code=proc.poll()
   if code is None:continue
   h.close();summary=ROOT/c["output"]/"test_summary.json"
   if code or not summary.is_file():failures.append({"cell_id":c["cell_id"],"exit_code":code})
   del active[group]
  if failures:
   for proc,h,_ in active.values():proc.terminate();h.close()
   break
 done=sum(1 for c in eligible if (ROOT/c["output"]/"test_summary.json").is_file())
 (root/"RECOVERY_STATUS.json").write_text(json.dumps({"registered":120,"complete":done,"failures":failures,"xor_unscored":20,"test_evaluated":done>0},indent=2,sort_keys=True)+"\n")
 if failures:raise RuntimeError(f"recovery failed closed: {failures}")
 return 0
if __name__=="__main__":raise SystemExit(main())
