#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t4_manifest.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(out,mh,arm):
 try:d=json.loads((out/"run_summary.json").read_text())
 except (OSError,json.JSONDecodeError):return False
 return d.get("manifest_sha256")==mh and d.get("arm")==arm and d.get("world_size")==4 and d.get("test_evaluated") is False
def main():
 m=json.loads(MP.read_text()); mh=sha(MP); cells=[(s,a,ROOT/f"outputs/t4/{a}/seed-{s}") for s in m["seeds"] for a in m["arms"]]; pending=[x for x in cells if not valid(x[2],mh,x[1])]; groups=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)]; (ROOT/"logs/t4").mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:8],pending[8:]; procs=[]
  for (seed,arm,out),gpus in zip(wave,groups):
   env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpus; log=(ROOT/f"logs/t4/{arm}-seed-{seed}.log").open("a"); cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/train_t4.py"),"--arm",arm,"--seed",str(seed),"--manifest",str(MP),"--manifest-sha256",mh,"--output",str(out)]; procs.append(((seed,arm,out),subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
  failed=[]
  for cell,proc,log in procs:
   code=proc.wait(); log.close()
   if code or not valid(cell[2],mh,cell[1]): failed.append((cell[:2],code))
  if failed: raise RuntimeError(f"T4 failed: {failed}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_t4.py")],cwd=ROOT).returncode
if __name__=="__main__": raise SystemExit(main())
