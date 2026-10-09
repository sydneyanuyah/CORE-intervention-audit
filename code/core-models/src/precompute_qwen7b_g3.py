#!/usr/bin/env python3
"""Cache selected Qwen hidden layers for development-only G3 selection."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer,BitsAndBytesConfig
LAYERS=(4,8,12,16,20,24,28)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--manifest-sha256",required=True);p.add_argument("--shard",type=int,required=True);a=p.parse_args()
 if sha(a.manifest)!=a.manifest_sha256 or not 0<=a.shard<32:raise ValueError("registry mismatch")
 m=json.loads(a.manifest.read_text());source=json.loads(Path(m["source_manifest"]).read_text());rows=[]
 for g in source["graphs"]:
  for w in json.loads(Path(g["files"]["worlds.json"]["path"]).read_text()):rows.append({"graph_seed":g["graph_seed"],"world_id":w["world_id"],"split":w["split"],"passage":w["passage"]})
 rows=rows[a.shard::32];tok=AutoTokenizer.from_pretrained(m["model_snapshot"],local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16);model=AutoModel.from_pretrained(m["model_snapshot"],local_files_only=True,quantization_config=q,device_map={"":0},dtype=torch.float16).eval();feat={k:[] for k in LAYERS}
 with torch.no_grad():
  for start in range(0,len(rows),4):
   b=tok([r["passage"] for r in rows[start:start+4]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0");o=model(**b,output_hidden_states=True);idx=b["attention_mask"].sum(1)-1
   for k in LAYERS:feat[k].extend(o.hidden_states[k][torch.arange(len(idx),device="cuda:0"),idx].float().cpu())
 out=Path(f"outputs/qwen7b-g3/cache/shard-{a.shard:02d}.pt");out.parent.mkdir(parents=True,exist_ok=True);torch.save({"manifest_sha256":a.manifest_sha256,"rows":rows,"features":{k:torch.stack(v).half() for k,v in feat.items()},"test_evaluated":False},out)
if __name__=="__main__":main()
