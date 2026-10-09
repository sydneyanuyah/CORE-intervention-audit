#!/usr/bin/env python3
"""Persistent 32-GPU owner for Qwen-7B and Phi-14B final-test evaluation."""
import hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; PY=str(ROOT/".venv/bin/python"); MODELS=("qwen7b","phi14b"); TASKS=("t2","t4","t5")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run_wave(jobs):
 pending=list(jobs); free=list(range(32)); active={}
 while pending or active:
  while pending and free:
   name,cmd,log=pending.pop(0); gpu=free.pop(0); env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); log.parent.mkdir(parents=True,exist_ok=True); stream=log.open("a"); active[gpu]=(name,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT),stream,log)
  time.sleep(1)
  for gpu,(name,proc,stream,log) in list(active.items()):
   code=proc.poll()
   if code is None: continue
   stream.close(); del active[gpu]; free.append(gpu); free.sort()
   if code:
    for _,other,s,_ in active.values(): other.terminate(); s.close()
    raise RuntimeError(f"failed {name}: {log}")
for model in MODELS: subprocess.run([PY,"scripts/prepare_expansion_final_test.py","--model-key",model],cwd=ROOT,check=True)
cache_jobs=[]
for model in MODELS:
 for task in TASKS:
  mp=ROOT/f"registry/{model}_{task}_final_test_manifest.json"; mh=sha(mp); manifest=json.loads(mp.read_text())
  for shard in range(manifest["feature_shards"]):
   output=ROOT/f"outputs/{model}-{task}-final-test/cache/shard-{shard:02d}.pt"
   if not output.exists(): cache_jobs.append((f"{model}-{task}-cache-{shard}",[PY,"src/precompute_expansion_test.py","--manifest",str(mp),"--manifest-sha256",mh,"--shard",str(shard)],ROOT/f"logs/expansion-final-test/{model}-{task}-cache-{shard:02d}.log"))
run_wave(cache_jobs)
score_jobs=[]
for model in MODELS:
 for task in TASKS:
  mp=ROOT/f"registry/{model}_{task}_final_test_manifest.json"; mh=sha(mp); manifest=json.loads(mp.read_text())
  for cell in manifest["source_cells"]:
   output=ROOT/f"outputs/{model}-{task}-final-test/{cell['method']}/seed-{cell['seed']}/test_summary.json"
   if not output.exists(): score_jobs.append((cell["cell_id"],[PY,"src/evaluate_expansion_test.py","--manifest",str(mp),"--manifest-sha256",mh,"--cell-id",cell["cell_id"]],ROOT/f"logs/expansion-final-test/{model}-{task}-{cell['method']}-{cell['seed']}.log"))
run_wave(score_jobs)
subprocess.run([PY,"scripts/report_expansion_final_test.py","--models",*MODELS],cwd=ROOT,check=True)
