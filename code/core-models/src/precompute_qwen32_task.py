#!/usr/bin/env python3
"""Extract one immutable shard of last-token Qwen2.5-32B features."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);p.add_argument("--shard",type=int,required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256:raise ValueError("manifest hash mismatch")
 m=json.loads(a.manifest.read_text());n=int(m["feature_shards"])
 if not 0<=a.shard<n or m["test_evaluated"] is not False or sha(m["data"])!=m["data_sha256"]:raise ValueError("contract mismatch")
 rows=[json.loads(x) for x in Path(m["data"]).read_text().splitlines() if x.strip()][a.shard::n]
 tok=AutoTokenizer.from_pretrained(m["model_snapshot"],local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="right"
 model=AutoModel.from_pretrained(m["model_snapshot"],local_files_only=True,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager",low_cpu_mem_usage=True).eval();features=[]
 with torch.no_grad():
  for start in range(0,len(rows),1):
   b=tok([r["text"] for r in rows[start:start+1]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0");h=model(**b).last_hidden_state;i=b["attention_mask"].sum(1)-1;features.extend(h[torch.arange(len(i),device=h.device),i].float().cpu())
 out=Path(f"outputs/qwen32-{m['task']}/cache/shard-{a.shard:02d}.pt");out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 torch.save({"manifest_sha256":a.manifest_sha256,"shard":a.shard,"indices":list(range(a.shard,a.shard+n*len(rows),n)),"features":torch.stack(features).half(),"test_evaluated":False},out)
if __name__=="__main__":main()
