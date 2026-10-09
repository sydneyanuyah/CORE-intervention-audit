#!/usr/bin/env python3
"""Pre-tokenize frozen G1 inputs once on CPU and publish an atomic cache."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=["ate","backadj","collider_bias","correlation","det-counterfactual","ett","exp_away","marginal","nde","nie"]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_rows(path,split):
 result=[]
 for line in Path(path).open():
  row=json.loads(line)
  if row.get("split")!=split or row.get("test_evaluated") is not False:raise ValueError("G1 cache split isolation failed")
  result.append(row)
 return result
def expand(rows):
 result=[]
 for row in rows:
  prompt=f"{row['given_info']} Question: {row['question']}"
  for command,candidate in enumerate(CANDIDATES):result.append((f"{prompt} Candidate answer: {candidate}",command))
 return result
def encode(tokenizer,items,chunk_size):
 ids=[];commands=[]
 for start in range(0,len(items),chunk_size):
  chunk=items[start:start+chunk_size];tokens=tokenizer([x[0] for x in chunk],padding=False,truncation=True,max_length=512)
  ids.extend(tokens["input_ids"]);commands.extend(x[1] for x in chunk)
 return {"dynamic":True,"pad_token_id":tokenizer.pad_token_id,"input_ids":ids,"command":commands}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,default=ROOT/"registry/g1_manifest.json");parser.add_argument("--output",type=Path,default=ROOT/"cache/g1-tokenized.pt");parser.add_argument("--chunk-size",type=int,default=256);args=parser.parse_args()
 manifest=json.loads(args.manifest.read_text())
 if sha(manifest["train_data"])!=manifest["train_data_sha256"] or sha(manifest["validation_data"])!=manifest["validation_data_sha256"]:raise ValueError("G1 source identity mismatch")
 tokenizer=AutoTokenizer.from_pretrained(manifest["model"])
 train=expand(source_rows(manifest["train_data"],"train"));validation=expand(source_rows(manifest["validation_data"],"validation"))
 payload={"version":1,"model":manifest["model"],"train_data_sha256":manifest["train_data_sha256"],"validation_data_sha256":manifest["validation_data_sha256"],"train":encode(tokenizer,train,args.chunk_size),"validation":encode(tokenizer,validation,args.chunk_size),"test_evaluated":False}
 args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp");torch.save(payload,temporary);os.replace(temporary,args.output)
 print(json.dumps({"output":str(args.output),"train_candidates":len(train),"validation_candidates":len(validation),"sha256":sha(args.output),"test_evaluated":False},sort_keys=True))
if __name__=="__main__":main()
