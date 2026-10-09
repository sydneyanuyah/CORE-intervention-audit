#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];MP=ROOT/"registry/g3_repaired_manifest.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(layer,mh):
 try:d=json.loads((ROOT/f"outputs/g3-repaired/layer-{layer}/run_summary.json").read_text())
 except (OSError,json.JSONDecodeError):return False
 return d.get("candidate_layer")==layer and d.get("manifest_sha256")==mh and d.get("world_size")==4 and d.get("test_evaluated") is False
def main():
 m=json.loads(MP.read_text());mh=sha(MP);pending=[x for x in m["candidate_layers"] if not valid(x,mh)];groups=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)];(ROOT/"logs/g3-repaired").mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:8],pending[8:];procs=[]
  for layer,gpus in zip(wave,groups):
   env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpus;log=(ROOT/f"logs/g3-repaired/layer-{layer}.log").open("a");cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/train_g3.py"),"--layer",str(layer),"--seed","531","--manifest",str(MP),"--manifest-sha256",mh,"--output",str(ROOT/f"outputs/g3-repaired/layer-{layer}")];procs.append((layer,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
  failed=[]
  for layer,proc,log in procs:
   code=proc.wait();log.close()
   if code or not valid(layer,mh):failed.append((layer,code))
  if failed:raise RuntimeError(f"G3 failed: {failed}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_g3.py")],cwd=ROOT).returncode
if __name__=="__main__":raise SystemExit(main())
