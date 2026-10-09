#!/usr/bin/env python3
"""Persistent resumable pipeline for every unfinished Qwen2.5-32B T/G transfer task."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TASKS=("t2","t3","t4","t5","g1","g2","g4")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_wave(jobs):
 pending=list(jobs);free=list(range(1));active={}
 while pending or active:
  while pending and free:
   _,cmd,log=pending.pop(0);gpu=free.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);f=log.open("a");active[gpu]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f,log)
  time.sleep(1)
  for gpu,(p,f,log) in list(active.items()):
   code=p.poll()
   if code is None:continue
   f.close();del active[gpu];free.append(gpu);free.sort()
   if code:raise RuntimeError(f"failed {log}")
def main():
 py=str(ROOT/".venv/bin/python");subprocess.run([py,"scripts/prepare_qwen32_remaining.py"],cwd=ROOT,check=True)
 for task in TASKS:
  mp=ROOT/f"registry/qwen32_{task}_manifest.json";mh=sha(mp);m=json.loads(mp.read_text());logs=ROOT/f"logs/qwen32-{task}";logs.mkdir(parents=True,exist_ok=True)
  pending=[]
  for i in range(1):
   out=ROOT/f"outputs/qwen32-{task}/cache/shard-{i:02d}.pt"
   if not out.exists():pending.append((i,[py,"src/precompute_qwen32_task.py","--manifest",str(mp),"--manifest-sha256",mh,"--shard",str(i)],logs/f"cache-{i:02d}.log"))
  if pending:run_wave(pending)
  cells=[]
  for c in m["cells"]:
   s=ROOT/c["output"]/"run_summary.json"
   try:r=json.loads(s.read_text());ok=r["cell_id"]==c["cell_id"] and r["manifest_sha256"]==mh and r["test_evaluated"] is False
   except Exception:ok=False
   if not ok:cells.append(c)
  if cells:
   jobs=[(0,[py,"src/train_qwen32_task.py","--cell-id",c["cell_id"],"--manifest",str(mp),"--manifest-sha256",mh],logs/f"{c['method']}-{c['seed']}.log") for c in cells]
   run_wave(jobs)
  summaries=[json.loads((ROOT/c["output"]/"run_summary.json").read_text()) for c in m["cells"]];report={"protocol":m["protocol"],"task":task,"completed_cells":len(summaries),"expected_cells":len(m["cells"]),"manifest_sha256":mh,"results":summaries,"evaluation_split":"validation","test_evaluated":False};(ROOT/f"reports/QWEN32_{task.upper()}.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"completed":list(TASKS),"test_evaluated":False}))
if __name__=="__main__":main()
