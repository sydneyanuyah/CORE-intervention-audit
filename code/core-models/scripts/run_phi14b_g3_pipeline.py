#!/usr/bin/env python3
"""Persistent development-only Phi layer selection."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LAYERS=(5,10,15,20,25,30,35,40)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wave(jobs):
 ps=[]
 for gpu,cmd,log in jobs:
  env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);f=log.open("a");ps.append((subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f,log))
 for p,f,l in ps:
  code=p.wait();f.close()
  if code:raise RuntimeError(f"failed {l}")
def main():
 py=str(ROOT/".venv/bin/python");src=ROOT/"registry/phi14b_f1_manifest.json";s=json.loads(src.read_text());m={"protocol":"phi4_14b_g3_development_layer_selection_v1","source_manifest":str(src),"source_manifest_sha256":sha(src),"model_snapshot":s["model_snapshot"],"layers":list(LAYERS),"seed":900,"selection":"maximum validation balanced accuracy; smallest-layer tie break","evaluation_split":"validation","test_evaluated":False};mp=ROOT/"registry/phi14b_g3_manifest.json";mp.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");mh=sha(mp);logs=ROOT/"logs/phi14b-g3";logs.mkdir(parents=True,exist_ok=True)
 jobs=[]
 for i in range(32):
  if not (ROOT/f"outputs/phi14b-g3/cache/shard-{i:02d}.pt").exists():jobs.append((i,[py,"src/precompute_phi14b_g3.py","--manifest",str(mp),"--manifest-sha256",mh,"--shard",str(i)],logs/f"cache-{i:02d}.log"))
 if jobs:wave(jobs)
 jobs=[]
 for gpu,k in enumerate(LAYERS):
  if not (ROOT/f"outputs/phi14b-g3/layer-{k}/run_summary.json").exists():jobs.append((gpu,[py,"src/train_phi14b_g3.py","--layer",str(k),"--manifest",str(mp),"--manifest-sha256",mh],logs/f"layer-{k}.log"))
 if jobs:wave(jobs)
 rows=[json.loads((ROOT/f"outputs/phi14b-g3/layer-{k}/run_summary.json").read_text()) for k in LAYERS];best=max(rows,key=lambda r:(r["validation_balanced_accuracy"],-r["layer"]));(ROOT/"reports/PHI14B_G3.json").write_text(json.dumps({"protocol":m["protocol"],"selected_layer":best["layer"],"selection_value":best["validation_balanced_accuracy"],"candidates":rows,"manifest_sha256":mh,"evaluation_split":"validation","test_evaluated":False},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
