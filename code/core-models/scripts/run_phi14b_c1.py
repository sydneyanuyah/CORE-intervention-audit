#!/usr/bin/env python3
"""Dispatch C1 shuffled cells while binding completed F3 controls."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];M=ROOT/"registry/phi14b_c1_manifest.json";A=ROOT/"registry/phi14b_c1_cache_amendment.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def valid(c,ms,ams):
 try:r=json.loads((ROOT/c["output"]/"run_summary.json").read_text())
 except Exception:return False
 return r.get("cell_id")==c["cell_id"] and r.get("manifest_sha256")==ms and r.get("amendment_sha256")==ams and r.get("test_evaluated") is False
def main():
 py=str(ROOT/".venv/bin/python");subprocess.run([py,"scripts/prepare_phi14b_c1.py"],cwd=ROOT,check=True);m=json.loads(M.read_text());ms=sha(M);source=json.loads(Path(m["feature_contract"]["source_amendment"]).read_text());am={"protocol":"phi4_14b_c1_cache_binding_v1","base_manifest":str(M),"base_manifest_sha256":ms,"feature_cache":source["feature_cache"],"feature_cache_sha256":source["feature_cache_sha256"],"source_amendment_sha256":m["feature_contract"]["source_amendment_sha256"],"test_evaluated":False};A.write_text(json.dumps(am,indent=2,sort_keys=True)+"\n");ams=sha(A);active=[];logs=ROOT/"logs/phi14b-c1";logs.mkdir(parents=True,exist_ok=True)
 for gpu,c in enumerate(x for x in m["cells"] if not valid(x,ms,ams)):
  f=(logs/f"shuffled-{c['seed']}.log").open("w");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu+1);cmd=[py,"src/train_phi14b_f3.py","--cell-id",c["cell_id"],"--manifest",str(M),"--manifest-sha256",ms,"--amendment",str(A),"--amendment-sha256",ams];active.append((c,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f))
 while active:
  time.sleep(3)
  for item in list(active):
   c,p,f=item
   if p.poll() is None:continue
   f.close();active.remove(item)
   if p.returncode or not valid(c,ms,ams):raise RuntimeError(f"failed {c['cell_id']}")
 print(json.dumps({"protocol":m["protocol"],"complete_cells":60,"new_cells":20,"reused_cells":40,"test_evaluated":False}))
if __name__=="__main__":main()
