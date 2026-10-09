#!/usr/bin/env python3
"""Run all 20 F2 XOR editor-zeroed C3 cells."""
from __future__ import annotations
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];GROUPS=tuple(",".join(map(str,range(s,s+4))) for s in range(0,32,4));CORE="${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py";CORE_SHA="not-published"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def cells():return [{"cell_id":f"c3:xor:edit_zeroed:{s}","seed":s,"graph_seed":s+2760,"reader":ROOT/f"outputs/f2/readers/xor/seed-{s}/reader.pt"} for s in range(301,321)]
def out(c):return ROOT/f"outputs/c3/xor/edit_zeroed/seed-{c['seed']}"
def valid(c):
 p=out(c)/"run_summary.json"
 if not p.is_file():return False
 try:d=load(p)
 except:return False
 return d.get("protocol")=="c3_xor_editor_zeroed_v1" and d.get("seed")==c["seed"] and d.get("graph_seed")==c["graph_seed"] and d.get("reader_sha256")==sha(c["reader"]) and d.get("world_size")==4 and d.get("test_evaluated") is False
def cmd(c):return [str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/evaluate_c3_xor_zeroed.py"),"--seed",str(c["seed"]),"--graph-seed",str(c["graph_seed"]),"--reader",str(c["reader"]),"--expected-reader-sha256",sha(c["reader"]),"--artifact-dir",f"${PRIVATE_STORAGE_ROOT}/CORE_f2/runs/graph_{c['graph_seed']}","--core-components",CORE,"--expected-core-sha256",CORE_SHA,"--output-dir",str(out(c))]
def main():
 logs=ROOT/"logs/c3-xor";logs.mkdir(parents=True,exist_ok=True);lock=(logs/"dispatcher.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cs=cells();active={}
 while True:
  for g,(p,s,c) in list(active.items()):
   code=p.poll()
   if code is None:continue
   s.close();del active[g]
   if code or not valid(c):raise RuntimeError(f"C3 XOR failed {c['cell_id']}")
  ids={v[2]["cell_id"] for v in active.values()};pending=[c for c in cs if not valid(c) and c["cell_id"] not in ids]
  for g in GROUPS:
   if g in active or not pending:continue
   c=pending.pop(0);s=(logs/(c["cell_id"].replace(":","_")+".log")).open("ab",buffering=0);p=subprocess.Popen(cmd(c),cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":g},stdout=s,stderr=subprocess.STDOUT);active[g]=(p,s,c)
  (logs/"status.json").write_text(json.dumps({"registered":20,"complete":sum(valid(c) for c in cs),"active":{g:v[2]["cell_id"] for g,v in active.items()},"test_evaluated":False},indent=2,sort_keys=True)+"\n")
  if not active and not pending:return 0
  time.sleep(5)
if __name__=="__main__":raise SystemExit(main())
