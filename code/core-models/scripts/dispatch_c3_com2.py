#!/usr/bin/env python3
"""Run the 20 registered repaired-Com2 C3 zeroed measurements."""
from __future__ import annotations
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; GROUPS=tuple(",".join(map(str,range(s,s+4))) for s in range(0,32,4))
H={"manifest":"not-published","cells":"not-published","catalog":"not-published","amendment":"not-published","queue":"not-published","truth":"not-published","provenance":"not-published","intervention-records":"not-published","counterfactual-records":"not-published","intervention-train":"not-published","counterfactual-train":"not-published","intervention-validation":"not-published","counterfactual-validation":"not-published","groups":"not-published"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def freeze():
 cat=load(ROOT/"registry/a2_source_catalog.json"); idx={(r["family"],r["seed"]):r for r in cat["cells"]}; rows=[]
 for seed in range(301,321):
  s=idx[("com2",2026090300+seed)]; reader=ROOT/s["source_directory"]/"best.pt"; op=ROOT/f"outputs/f2-fixed/com2/o3/seed-{seed}/operator.pt"
  if sha(reader)!=s["checkpoint_sha256"] or not op.is_file():raise RuntimeError("C3 source hash/path failure")
  rows.append({"cell_id":f"c3:com2:edit_zeroed:{seed}","seed":seed,"reader":str(reader.relative_to(ROOT)),"reader_sha256":sha(reader),"f2_operator":str(op.relative_to(ROOT)),"f2_operator_sha256":sha(op),"test_evaluated":False})
 p=ROOT/"registry/c3_com2_cells.json"; text=json.dumps({"protocol":"c3_com2_cells_v1","cell_count":20,"cells":rows,"test_evaluated":False},indent=2,sort_keys=True)+"\n"
 if p.exists() and p.read_text()!=text:raise RuntimeError("refusing to mutate C3 Com2 registry")
 p.write_text(text);return rows
def out(c):return ROOT/f"outputs/c3/com2/edit_zeroed/seed-{c['seed']}"
def valid(c):
 p=out(c)/"run_summary.json"
 if not p.is_file():return False
 try:d=load(p)
 except: return False
 return d.get("protocol")=="c3_com2_editor_zeroed_v1" and d.get("seed")==c["seed"] and d.get("world_size")==4 and d.get("checkpoint_sha256")==c["reader_sha256"] and d.get("f2_operator_sha256")==c["f2_operator_sha256"] and d.get("test_evaluated") is False
def cmd(c):
 source=(ROOT/c["reader"]).parent
 x=[str(ROOT/".venv/bin/torchrun"),"--standalone","--nproc_per_node=4",str(ROOT/"src/evaluate_c3_com2_zeroed.py"),"--seed",str(c["seed"]),"--f2-operator",str(ROOT/c["f2_operator"]),"--expected-f2-operator-sha256",c["f2_operator_sha256"],"--checkpoint",str(ROOT/c["reader"]),"--expected-checkpoint-sha256",c["reader_sha256"],"--tokenizer",str(source/"tokenizer"),"--data-root",str(ROOT/"data"),"--manifest",str(ROOT/"registry/f2_manifest.json"),"--expected-manifest-sha256",H["manifest"],"--cells",str(ROOT/"registry/f2_cells.json"),"--expected-cells-sha256",H["cells"],"--source-catalog",str(ROOT/"registry/a2_source_catalog.json"),"--expected-source-catalog-sha256",H["catalog"],"--amendment",str(ROOT/"registry/f2_com2_execution_amendment_v2.json"),"--expected-amendment-sha256",H["amendment"],"--queue",str(ROOT/"data/two_edit/manual/annotation_queue.jsonl"),"--expected-queue-sha256",H["queue"],"--truth",str(ROOT/"data/two_edit/manual/authoritative_manual_truth.jsonl"),"--expected-truth-sha256",H["truth"],"--provenance",str(ROOT/"data/two_edit/manual/authoritative_manual_truth.provenance.json"),"--expected-provenance-sha256",H["provenance"],"--output-dir",str(out(c))]
 for k in H:
  if k not in {"manifest","cells","catalog","amendment","queue","truth","provenance"}:x += [f"--expected-{k}-sha256",H[k]]
 return x
def main():
 logs=ROOT/"logs/c3-com2";logs.mkdir(parents=True,exist_ok=True);lock=(logs/"dispatcher.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cells=freeze();active={}
 while True:
  for g,(p,s,c) in list(active.items()):
   code=p.poll()
   if code is None:continue
   s.close();del active[g]
   if code or not valid(c):raise RuntimeError(f"C3 cell failed {c['cell_id']}")
  ids={v[2]["cell_id"] for v in active.values()};pending=[c for c in cells if not valid(c) and c["cell_id"] not in ids]
  for g in GROUPS:
   if g in active or not pending:continue
   c=pending.pop(0);s=(logs/(c["cell_id"].replace(":","_")+".log")).open("ab",buffering=0);p=subprocess.Popen(cmd(c),cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":g},stdout=s,stderr=subprocess.STDOUT);active[g]=(p,s,c)
  (logs/"status.json").write_text(json.dumps({"registered":20,"complete":sum(valid(c) for c in cells),"active":{g:v[2]["cell_id"] for g,v in active.items()},"test_evaluated":False},indent=2,sort_keys=True)+"\n")
  if not active and not pending:return 0
  time.sleep(5)
if __name__=="__main__":raise SystemExit(main())
