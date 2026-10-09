#!/usr/bin/env python3
"""Continuously occupy all GPUs while materializing independent Phi feature caches."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[]
base=[
 ("f1","src/precompute_llama70b_f1.py","registry/llama70b_f1_manifest.json"),
 ("f2","src/precompute_llama70b_cladder.py","registry/llama70b_cladder_extension_manifest.json"),
 ("f3","src/precompute_llama70b_f1.py","registry/llama70b_f3_manifest.json"),
 ("f4","src/precompute_llama70b_f1.py","registry/llama70b_f4_manifest.json"),
 ("a1","src/precompute_llama70b_f1.py","registry/llama70b_a1_manifest.json"),
]
for name,script,manifest in base:
 m=R/manifest;d=json.loads(m.read_text())
 if not Path(d["feature_cache_summary"]).is_file():jobs.append((name,[PY,script,"--manifest",str(m),"--manifest-sha256",sha(m)]))
for task in ("t2","t3","t4","t5","g1","g2","g4"):
 m=R/f"registry/llama70b_{task}_manifest.json";mh=sha(m)
 for shard in range(1):
  if not (R/f"outputs/llama70b-{task}/cache/shard-{shard:02d}.pt").is_file():jobs.append((f"{task}-cache-{shard:02d}",[PY,"src/precompute_llama70b_task.py","--manifest",str(m),"--manifest-sha256",mh,"--shard",str(shard)]))
logs=R/"logs/llama70b-cache";logs.mkdir(parents=True,exist_ok=True);free=list(range(1));active={}
while jobs or active:
 while jobs and free:
  gpu=free.pop(0);name,cmd=jobs.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);stream=(logs/f"{name}.log").open("a");active[gpu]=(name,subprocess.Popen(cmd,cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT),stream)
 time.sleep(2)
 for gpu,(name,proc,stream) in list(active.items()):
  code=proc.poll()
  if code is None:continue
  stream.close();del active[gpu];free.append(gpu);free.sort()
  if code:
   for _,other,s in active.values():other.terminate();s.close()
   raise RuntimeError(f"Phi cache job failed: {name}")
