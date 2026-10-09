#!/usr/bin/env python3
"""Prepare, cache, freeze, and dispatch all registered Qwen-7B F3 cells."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];M=ROOT/"registry/qwen7b_f3_manifest.json";A=ROOT/"registry/qwen7b_f3_cache_amendment.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(c,ms,ams):
 try:r=json.loads((ROOT/c["output"]/"run_summary.json").read_text())
 except Exception:return False
 return r.get("cell_id")==c["cell_id"] and r.get("manifest_sha256")==ms and r.get("amendment_sha256")==ams and r.get("gpu_count")==1 and r.get("evaluation_split")=="validation" and r.get("test_evaluated") is False
def main():
 py=str(ROOT/".venv/bin/python");subprocess.run([py,"scripts/prepare_qwen7b_f3.py"],cwd=ROOT,check=True);ms=sha(M);m=json.loads(M.read_text())
 if not Path(m["feature_cache_summary"]).exists():subprocess.run([py,"src/precompute_qwen7b_f1.py","--manifest",str(M),"--manifest-sha256",ms],cwd=ROOT,check=True)
 summary=json.loads(Path(m["feature_cache_summary"]).read_text());am={"protocol":"qwen2p5_7b_f3_cache_binding_v1","base_manifest":str(M),"base_manifest_sha256":ms,"feature_cache":m["feature_cache"],"feature_cache_sha256":summary["feature_cache_sha256"],"test_evaluated":False};A.write_text(json.dumps(am,indent=2,sort_keys=True)+"\n");ams=sha(A)
 pending=[c for c in m["cells"] if not valid(c,ms,ams)];logs=ROOT/"logs/qwen7b-f3";logs.mkdir(parents=True,exist_ok=True)
 while pending:
  wave,pending=pending[:32],pending[32:];active=[]
  for gpu,c in enumerate(wave):
   out=ROOT/c["output"]
   if out.exists() and not valid(c,ms,ams):raise RuntimeError(f"partial {out}")
   f=(logs/f"{c['condition']}-{c['method']}-{c['seed']}.log").open("w");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);cmd=[py,"src/train_qwen7b_f3.py","--cell-id",c["cell_id"],"--manifest",str(M),"--manifest-sha256",ms,"--amendment",str(A),"--amendment-sha256",ams];active.append((c,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f))
  while active:
   time.sleep(3)
   for item in list(active):
    c,p,f=item
    if p.poll() is None:continue
    f.close();active.remove(item)
    if p.returncode or not valid(c,ms,ams):
     for _,q,z in active:q.terminate();z.close()
     raise RuntimeError(f"failed {c['cell_id']}")
 print(json.dumps({"protocol":m["protocol"],"complete_cells":len(m["cells"]),"test_evaluated":False}))
if __name__=="__main__":main()
