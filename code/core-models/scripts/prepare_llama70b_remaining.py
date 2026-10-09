#!/usr/bin/env python3
"""Freeze normalized train/validation registries for the remaining Llama-3.3-70B tasks."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path("${PRIVATE_STORAGE_ROOT}/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/not-published")
TASKS=("t2","t3","t4","t5","g1","g2","g4")
SEEDS={"t2":range(701,721),"t3":range(721,741),"t4":range(741,761),"t5":range(761,781),"g1":range(781,801),"g2":range(801,821),"g4":range(821,841)}
METHODS={t:("baseline","o3") for t in TASKS}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in rows))
def heldout(group):return int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0
def directions(row):
 keys=sorted(row["factual_state"]);out=[]
 for k in keys:
  a,b=float(row["factual_state"][k]),float(row["intervened_state"][k]);out.append(0 if b<a else 2 if b>a else 1)
 return out
def options(r):
 x=r.get("options",[])
 if isinstance(x,list):return [str(v) for v in x]
 return [v.strip() for v in re.findall(r"(?:^|\n)\s*[-*]?\s*(?:[A-F]|\d+)[\).]\s*(.*?)(?=(?:\n\s*[-*]?\s*(?:[A-F]|\d+)[\).])|$)",x,re.S)]
def normalize(task):
 d=ROOT/"data/experiments"
 if task=="t2":
  src=lines(d/task/"csuite_pairs.jsonl");return [{"id":r["id"],"group":r["graph_group_id"],"split":"validation" if r["generation_seed"]==505 else "train","text":r["rendered_input"],"label":directions(r),"command":sorted(r["factual_state"]).index(r["intervention"]["target"]),"source":r["sem_family"],"test_evaluated":False} for r in src]
 if task=="t4":
  src=lines(d/task/"csuite_rung_views.jsonl");arms={"changing_only":0,"imagining_only":1,"joint":2};return [{"id":r["id"],"group":r["graph_group_id"],"split":"validation" if r["generation_seed"]==515 else "train","text":r["rendered_input"],"label":directions(r),"command":arms[r["arm"]],"source":r["arm"],"test_evaluated":False} for r in src]
 if task=="t3":
  out=[]
  for r in lines(d/task/"corr2cause_pairs.jsonl"):
   split="validation" if heldout(r["graph_group_id"]) else "train"
   for variant in ("clean","paraphrase","refactorization"):
    out.append({"id":f"{r['pair_id']}:{variant}","pair_id":r["pair_id"],"group":r["graph_group_id"],"split":split,"variant":variant,"text":r[variant]["input"],"label":int(r["label"]),"command":int(hashlib.sha256(r["template"].encode()).hexdigest()[:4],16)%256,"source":r["template"],"test_evaluated":False})
  return out
 if task=="g2":
  out=[]
  for r in lines(d/task/"sequences_validation.jsonl"):
   split="validation" if heldout(r["graph_group_id"]) else "train";g=r["graph"];text=f"Graph parents: {json.dumps(g['parents'],sort_keys=True)}. Factual: {json.dumps(r['factual_state'],sort_keys=True)}. Ordered edits: {json.dumps(r['ordered_edits'],sort_keys=True)}"
   out.append({"id":r["id"],"group":r["graph_group_id"],"split":split,"text":text,"label":[int(r["gold_final_state"][k]) for k in sorted(r["gold_final_state"])],"command":sorted(r["gold_final_state"]).index(r["target"]),"source":"last_write_wins","test_evaluated":False})
  return out
 if task=="g1":
  src=lines(d/task/"estimand_train.jsonl")+lines(d/task/"estimand_validation.jsonl");labels=sorted({r["query_type"] for r in src});return [{"id":r["id"],"group":r.get("story_id",r["id"]),"split":r["split"],"text":f"{r['given_info']} Question: {r['question']}","label":labels.index(r["query_type"]),"command":labels.index(r["query_type"]),"source":r["query_type"],"test_evaluated":False} for r in src]
 if task=="g4":
  src=lines(d/task/"evidence_inference/adjudicated_prompts.jsonl");labels=["decreased","no_change","increased"];return [{"id":r["id"],"group":r["article_group_id"],"split":r["split"],"text":f"Intervention: {r['intervention']}. Comparator: {r['comparator']}. Outcome: {r['outcome']}.","label":labels.index(r["label"]),"command":labels.index(r["label"]),"source":"evidence_inference","test_evaluated":False} for r in src]
 if task=="t5":
  src=lines(d/task/"native_train.jsonl")+lines(d/task/"native_validation.jsonl");out=[]
  for r in src:
   if r["source"]=="com2":continue
   prompt=r.get("input") or " ".join(str(r.get(k,"")) for k in ("premise","context","question"));answer=str(r["answer"])
   if r["source"]=="counterbench":opts=["no","yes"];answer=answer.lower()
   elif r["source"]=="corr2cause":opts=["0","1"]
   else:opts=options(r)
   if answer not in opts and r["source"]=="crass":opts.append(answer)
   for i,opt in enumerate(opts):out.append({"id":f"{r['id']}:{i}","item_id":r["id"],"group":r.get("group_id",r["id"]),"split":r["split"],"text":f"{prompt} Candidate answer: {opt}","label":int(opt==answer or str(i)==answer),"command":i,"source":r["source"],"test_evaluated":False})
  return out
 raise ValueError(task)
def main():
 if not (MODEL/"config.json").is_file():raise FileNotFoundError(MODEL)
 for task in TASKS:
  rows=normalize(task)
  if not rows or {r["split"] for r in rows}!={"train","validation"} or any(r["test_evaluated"] is not False for r in rows):raise ValueError(f"bad split contract {task}")
  data=ROOT/f"data/llama70b/{task}.jsonl";write(data,rows)
  cells=[{"cell_id":f"llama70b-{task}:{method}:{seed}","method":method,"seed":seed,"output":f"outputs/llama70b-{task}/{method}/seed-{seed}","gpu_count":1} for seed in SEEDS[task] for method in METHODS[task]]
  m={"protocol":f"llama3_3_70b_{task}_frozen_feature_v1","task":task,"model":"meta-llama/Llama-3.3-70B-Instruct","model_revision":MODEL.name,"model_snapshot":str(MODEL),"data":str(data),"data_sha256":sha(data),"train_rows":sum(r["split"]=="train" for r in rows),"validation_rows":sum(r["split"]=="validation" for r in rows),"methods":list(METHODS[task]),"seeds":list(SEEDS[task]),"cells":cells,"feature_shards":1,"backbone_frozen":True,"evaluation_split":"validation","test_evaluated":False}
  p=ROOT/f"registry/llama70b_{task}_manifest.json";p.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(task,len(rows),sha(p))
if __name__=="__main__":main()
