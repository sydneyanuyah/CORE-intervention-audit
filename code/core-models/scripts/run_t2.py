#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t2_manifest.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(path,manifest_sha):
 try:d=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return False
 return d.get("manifest_sha256")==manifest_sha and d.get("world_size")==4 and d.get("test_evaluated") is False
def main():
 m=json.loads(MP.read_text()); mh=sha(MP); (ROOT/"logs/t2").mkdir(parents=True,exist_ok=True); procs=[]
 groups=["0,1,2,3","4,5,6,7","8,9,10,11","12,13,14,15","16,17,18,19"]
 for seed,gpus in zip(m["seeds"],groups):
  out=ROOT/f"outputs/t2/seed-{seed}"
  if valid(out/"run_summary.json",mh): continue
  env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpus
  log=(ROOT/f"logs/t2/seed-{seed}.log").open("a")
  cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/train_t2.py"),"--seed",str(seed),"--manifest",str(MP),"--manifest-sha256",mh,"--output",str(out)]
  procs.append((seed,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
 failed=[]
 for seed,proc,log in procs:
  code=proc.wait(); log.close()
  if code or not valid(ROOT/f"outputs/t2/seed-{seed}/run_summary.json",mh): failed.append((seed,code))
 if failed: raise RuntimeError(f"T2 failed cells: {failed}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_t2.py")],cwd=ROOT).returncode
if __name__=="__main__": raise SystemExit(main())
