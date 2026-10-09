#!/usr/bin/env python3
"""Dispatch all registered Phi-14B F1 cells over free GPUs."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];M=ROOT/"registry/phi14b_f1_manifest.json";A=ROOT/"registry/phi14b_f1_cache_amendment.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(c,ms,ams):
 try:r=json.loads((ROOT/c["output"]/"run_summary.json").read_text())
 except Exception:return False
 return r.get("cell_id")==c["cell_id"] and r.get("manifest_sha256")==ms and r.get("amendment_sha256")==ams and r.get("test_evaluated") is False
def main():
 m=json.loads(M.read_text());ms=sha(M);ams=sha(A);pending=[c for c in m["cells"] if not valid(c,ms,ams)];logs=ROOT/"logs/phi14b-f1";logs.mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:32],pending[32:];active=[]
  for gpu,c in enumerate(wave):
   out=ROOT/c["output"]
   if out.exists() and not valid(c,ms,ams):raise RuntimeError(f"partial {out}")
   f=(logs/f"{c['method']}-{c['seed']}.log").open("w");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);cmd=[str(ROOT/".venv/bin/python"),str(ROOT/"src/train_phi14b_f1.py"),"--cell-id",c["cell_id"],"--manifest",str(M),"--manifest-sha256",ms,"--amendment",str(A),"--amendment-sha256",ams];active.append((c,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f))
  while active:
   time.sleep(3)
   for item in list(active):
    c,p,f=item
    if p.poll() is None:continue
    f.close();active.remove(item)
    if p.returncode or not valid(c,ms,ams):
     for _,q,z in active:q.terminate();z.close()
     raise RuntimeError(f"failed {c['cell_id']}")
if __name__=="__main__":main()
