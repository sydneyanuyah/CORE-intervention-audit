#!/usr/bin/env python3
"""Cache frozen Phi features for all registered F1 XOR worlds."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer,BitsAndBytesConfig
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--manifest-sha256",required=True);a=ap.parse_args()
 if sha(a.manifest)!=a.manifest_sha256: raise ValueError("manifest hash mismatch")
 m=json.loads(a.manifest.read_text()); rows=[]
 for g in m["graphs"]:
  wpath=Path(g["files"]["worlds.json"]["path"])
  if sha(wpath)!=g["files"]["worlds.json"]["sha256"]: raise ValueError("world hash mismatch")
  for w in json.loads(wpath.read_text()):
   if w["split"] not in {"train","validation"}: raise ValueError("forbidden split")
   rows.append({"graph_seed":g["graph_seed"],"world_id":w["world_id"],"split":w["split"],"passage":w["passage"]})
 tok=AutoTokenizer.from_pretrained(m["model_snapshot"],local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="right"
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
 model=AutoModel.from_pretrained(m["model_snapshot"],local_files_only=True,quantization_config=q,device_map={"":0},dtype=torch.float16).eval(); feats=[]
 with torch.no_grad():
  for start in range(0,len(rows),4):
   prompts=[f"Causal system description:\n{x['passage']}\nEncode this factual world for downstream intervention editing." for x in rows[start:start+4]]
   b=tok(prompts,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0");h=model(**b).last_hidden_state;idx=b["attention_mask"].sum(1)-1;feats.extend(h[torch.arange(len(idx),device=h.device),idx].float().cpu())
 out=Path(m["feature_cache"]);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise FileExistsError(out)
 torch.save({"protocol":m["protocol"],"manifest_sha256":a.manifest_sha256,"hidden_size":int(model.config.hidden_size),"records":rows,"features":torch.stack(feats).half(),"test_evaluated":False},out)
 s={"manifest_sha256":a.manifest_sha256,"feature_cache_sha256":sha(out),"world_count":len(rows),"train_worlds":sum(x["split"]=="train" for x in rows),"validation_worlds":sum(x["split"]=="validation" for x in rows),"hidden_size":int(model.config.hidden_size),"test_evaluated":False};Path(m["feature_cache_summary"]).write_text(json.dumps(s,indent=2,sort_keys=True)+"\n");print(json.dumps(s))
if __name__=="__main__": main()
