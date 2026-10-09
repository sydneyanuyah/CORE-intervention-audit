#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(task):
 mp=ROOT/f"registry/{task}_manifest.json";m=json.loads(mp.read_text());mh=sha(mp);cells=[(method,seed,ROOT/f"outputs/{task}/{method}/seed-{seed}") for seed in m["seeds"] for method in m["methods"]]
 def valid(x):
  method,seed,out=x
  try:d=json.loads((out/"run_summary.json").read_text())
  except (OSError,json.JSONDecodeError):return False
  return d.get("method")==method and d.get("seed")==seed and d.get("world_size")==4 and d.get("manifest_sha256")==mh and d.get("test_evaluated") is False
 pending=[x for x in cells if not valid(x)];groups=[f"{i},{i+1},{i+2},{i+3}" for i in range(0,32,4)];(ROOT/f"logs/{task}").mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:8],pending[8:];procs=[]
  for cell,gpus in zip(wave,groups):
   method,seed,out=cell;env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpus;log=(ROOT/f"logs/{task}/{method}-seed-{seed}.log").open("a");cmd=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/train_candidate_transfer.py"),"--task",task,"--method",method,"--seed",str(seed),"--manifest",str(mp),"--manifest-sha256",mh,"--output",str(out)]
   token_cache=ROOT/"cache/g1-tokenized.pt"
   if task=="g1" and token_cache.is_file():cmd.extend(["--token-cache",str(token_cache)])
   procs.append((cell,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
  failed=[]
  for cell,proc,log in procs:
   code=proc.wait();log.close()
   if code or not valid(cell):failed.append((cell[:2],code))
  if failed:raise RuntimeError(f"{task.upper()} failed: {failed}")
 return subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/report_candidate_transfer.py"),"--task",task],cwd=ROOT).returncode
if __name__=="__main__":raise SystemExit(main(sys.argv[1]))
