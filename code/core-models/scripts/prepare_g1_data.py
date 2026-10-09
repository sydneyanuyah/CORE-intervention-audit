#!/usr/bin/env python3
"""Create story-disjoint train/validation G1 estimand-selection records."""
from __future__ import annotations
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path("${USER_HOME}/Documents/core-data/data/raw/cladder/cladder-3d2d1169.tar.gz")
MEMBER="cladder-not-published/data/cladder-v1-questions.json"
def split(story):return "validation" if int.from_bytes(hashlib.sha256(story.encode()).digest()[:2],"big")%5==0 else "train"
def main():
 with tarfile.open(ARCHIVE) as tf: rows=json.load(tf.extractfile(MEMBER))
 out=ROOT/"data/experiments/g1";out.mkdir(parents=True,exist_ok=True); handles={s:(out/f"estimand_{s}.jsonl").open("w") for s in ("train","validation")}
 try:
  for r in rows:
   meta=r["meta"]; s=split(meta["story_id"]); item={"id":f"cladder-g1-{r['question_id']}","story_id":meta["story_id"],"graph_id":meta["graph_id"],"given_info":r["given_info"],"question":r["question"],"query_type":meta["query_type"],"estimand":meta.get("estimand"),"formal_form":meta.get("formal_form"),"split":s,"source":"cladder","test_evaluated":False};handles[s].write(json.dumps(item,sort_keys=True)+"\n")
 finally:
  for h in handles.values():h.close()
if __name__=="__main__":main()
