#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/g2_manifest.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(c,mh):
 p=ROOT/c["output"]/"measurement_summary.json"
 try:d=json.loads(p.read_text())
 except Exception:return False
 return d.get("test_evaluated") is False and d.get("world_size")==4 and d.get("manifest_sha256")==mh and d.get("checkpoint_sha256")==c["checkpoint_sha256"]
def main():
 m=json.loads(MP.read_text()); mh=sha(MP); pending=[c for c in m["cells"] if not valid(c,mh)]; groups=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)]; (ROOT/"logs/g2").mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:8],pending[8:]; procs=[]
  for c,g in zip(wave,groups):
   env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=g;log=(ROOT/f"logs/g2/{c['method']}-seed-{c['seed']}.log").open("a");cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/evaluate_g2.py"),"--method",c["method"],"--seed",str(c["seed"]),"--manifest",str(MP),"--manifest-sha256",mh,"--output",str(ROOT/c["output"])];procs.append((c,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
  bad=[]
  for c,p,l in procs:
   code=p.wait();l.close()
   if code or not valid(c,mh):bad.append((c["cell_id"],code))
  if bad:raise RuntimeError(f"G2 failed cells: {bad}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_g2.py")],cwd=ROOT).returncode
if __name__=="__main__":raise SystemExit(main())
