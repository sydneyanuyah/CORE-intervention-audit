#!/usr/bin/env python3
"""Persistent resumable L1/L2/L3 Qwen-7B law pipeline."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 py=str(ROOT/".venv/bin/python");subprocess.run([py,"scripts/prepare_qwen7b_laws.py"],cwd=ROOT,check=True)
 for task in ("l1","l2","l3"):
  mp=ROOT/f"registry/qwen7b_{task}_manifest.json";mh=sha(mp);m=json.loads(mp.read_text());pending=[];logs=ROOT/f"logs/qwen7b-{task}";logs.mkdir(parents=True,exist_ok=True)
  for c in m["cells"]:
   try:r=json.loads((ROOT/c["output"]/"run_summary.json").read_text());ok=r["cell_id"]==c["cell_id"] and r["manifest_sha256"]==mh and r["test_evaluated"] is False
   except Exception:ok=False
   if not ok:pending.append(c)
  free=list(range(32));active={}
  while pending or active:
   while pending and free:
    gpu=free.pop(0);c=pending.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);f=(logs/f"{c['method']}-{c['seed']}.log").open("a");cmd=[py,"src/train_qwen7b_law_cell.py","--cell-id",c["cell_id"],"--manifest",str(mp),"--manifest-sha256",mh];active[gpu]=(c,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f)
   time.sleep(1)
   for gpu,(c,p,f) in list(active.items()):
    code=p.poll()
    if code is None:continue
    f.close();del active[gpu];free.append(gpu);free.sort()
    if code:raise RuntimeError(f"failed {c['cell_id']}")
  rr=[json.loads((ROOT/c["output"]/"run_summary.json").read_text()) for c in m["cells"]];(ROOT/f"reports/QWEN7B_{task.upper()}.json").write_text(json.dumps({"protocol":m["protocol"],"completed_cells":len(rr),"expected_cells":len(m["cells"]),"manifest_sha256":mh,"results":rr,"evaluation_split":"validation","test_evaluated":False},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
