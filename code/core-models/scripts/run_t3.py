#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t3_manifest.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(cell,mh):
 p=ROOT/cell["output"]/"measurement_summary.json"
 try:d=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return False
 return d.get("test_evaluated") is False and d.get("checkpoint_sha256")==cell["checkpoint_sha256"] and d.get("distributed",{}).get("world_size")==4 and (ROOT/cell["output"]/"predictions.jsonl").is_file()
def main():
 m=json.loads(MP.read_text()); mh=sha(MP); pending=[c for c in m["cells"] if not valid(c,mh)]; groups=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)]; (ROOT/"logs/t3").mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:8],pending[8:]; procs=[]
  for cell,gpus in zip(wave,groups):
   env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpus; log=(ROOT/f"logs/t3/{cell['view']}-seed-{cell['seed']}.log").open("a")
   cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/evaluate_t3.py"),"--cell-id",cell["cell_id"],"--manifest",str(MP),"--manifest-sha256",mh,"--output",str(ROOT/cell["output"])]
   procs.append((cell,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
  failed=[]
  for cell,proc,log in procs:
   code=proc.wait(); log.close()
   if code or not valid(cell,mh): failed.append((cell["cell_id"],code))
  if failed: raise RuntimeError(f"T3 failed cells: {failed}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_t3.py")],cwd=ROOT).returncode
if __name__=="__main__": raise SystemExit(main())
