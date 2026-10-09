#!/usr/bin/env python3
"""Persistent development-only Phi layer selection."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LAYERS=(5,10,15,20,25,30,35,40)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wave(jobs):
 pending=list(jobs);free=[0,1];active={}
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
 py=str(ROOT/".venv/bin/python");src=ROOT/"registry/llama70b_f1_manifest.json";s=json.loads(src.read_text());m={"protocol":"llama3_3_70b_g3_development_layer_selection_v1","source_manifest":str(src),"source_manifest_sha256":sha(src),"model_snapshot":s["model_snapshot"],"layers":list(LAYERS),"seed":900,"selection":"maximum validation balanced accuracy; smallest-layer tie break","evaluation_split":"validation","test_evaluated":False};mp=ROOT/"registry/llama70b_g3_manifest.json";mp.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");mh=sha(mp);logs=ROOT/"logs/llama70b-g3";logs.mkdir(parents=True,exist_ok=True)
 jobs=[]
 for i in range(1):
  if not (ROOT/f"outputs/llama70b-g3/cache/shard-{i:02d}.pt").exists():jobs.append((i,[py,"src/precompute_llama70b_g3.py","--manifest",str(mp),"--manifest-sha256",mh,"--shard",str(i)],logs/f"cache-{i:02d}.log"))
 if jobs:
  # Cache extraction loads one 70B backbone sharded over both H100s.
  for _,cmd,log in jobs:
   with log.open("a") as f:
    subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 jobs=[]
 for gpu,k in enumerate(LAYERS):
  if not (ROOT/f"outputs/llama70b-g3/layer-{k}/run_summary.json").exists():jobs.append((gpu,[py,"src/train_llama70b_g3.py","--layer",str(k),"--manifest",str(mp),"--manifest-sha256",mh],logs/f"layer-{k}.log"))
 if jobs:wave(jobs)
 rows=[json.loads((ROOT/f"outputs/llama70b-g3/layer-{k}/run_summary.json").read_text()) for k in LAYERS];best=max(rows,key=lambda r:(r["validation_balanced_accuracy"],-r["layer"]));(ROOT/"reports/LLAMA70B_G3.json").write_text(json.dumps({"protocol":m["protocol"],"selected_layer":best["layer"],"selection_value":best["validation_balanced_accuracy"],"candidates":rows,"manifest_sha256":mh,"evaluation_split":"validation","test_evaluated":False},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
