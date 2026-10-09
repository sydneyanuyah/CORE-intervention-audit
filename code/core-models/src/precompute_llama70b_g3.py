#!/usr/bin/env python3
"""Cache selected Phi hidden layers for development-only G3 selection."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer
LAYERS=(5,10,15,20,25,30,35,40)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);p.add_argument("--shard",type=int,required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256 or not 0<=a.shard<1:raise ValueError("registry mismatch")
 m=json.loads(a.manifest.read_text());source=json.loads(Path(m["source_manifest"]).read_text());rows=[]
 for g in source["graphs"]:
  for w in json.loads(Path(g["files"]["worlds.json"]["path"]).read_text()):rows.append({"graph_seed":g["graph_seed"],"world_id":w["world_id"],"split":w["split"],"passage":w["passage"]})
 rows=rows[a.shard::1];tok=AutoTokenizer.from_pretrained(m["model_snapshot"],local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModel.from_pretrained(m["model_snapshot"],local_files_only=True,dtype=torch.bfloat16,device_map="balanced",max_memory={0:"76GiB",1:"76GiB"},attn_implementation="eager",low_cpu_mem_usage=True).eval();feat={k:[] for k in LAYERS}
 with torch.no_grad():
  for start in range(0,len(rows),1):
   b=tok([r["passage"] for r in rows[start:start+1]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0");o=model(**b,output_hidden_states=True);idx=b["attention_mask"].sum(1)-1
   for k in LAYERS:
    layer=o.hidden_states[k];local_idx=idx.to(layer.device)
    feat[k].extend(layer[torch.arange(len(local_idx),device=layer.device),local_idx].float().cpu())
 out=Path(f"outputs/llama70b-g3/cache/shard-{a.shard:02d}.pt");out.parent.mkdir(parents=True,exist_ok=True);torch.save({"manifest_sha256":a.manifest_sha256,"rows":rows,"features":{k:torch.stack(v).half() for k,v in feat.items()},"test_evaluated":False},out)
if __name__=="__main__":main()
