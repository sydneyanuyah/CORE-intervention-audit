#!/usr/bin/env python3
"""Extract a registered shard of final-test features without fitting anything."""
import argparse, hashlib, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--manifest-sha256",required=True); parser.add_argument("--shard",type=int,required=True); args=parser.parse_args()
    if sha(args.manifest)!=args.manifest_sha256: raise ValueError("manifest hash mismatch")
    manifest=json.loads(args.manifest.read_text()); shards=int(manifest["feature_shards"])
    if manifest.get("authorization")!="EXPANSION_FINAL_ONE_SHOT" or manifest.get("test_evaluated") is not True: raise PermissionError("not an authorized final-test manifest")
    if not 0<=args.shard<shards or sha(manifest["data"])!=manifest["data_sha256"]: raise ValueError("data contract mismatch")
    all_rows=[json.loads(line) for line in Path(manifest["data"]).read_text().splitlines() if line.strip()]
    selected=all_rows[args.shard::shards]; tokenizer=AutoTokenizer.from_pretrained(manifest["model_snapshot"],local_files_only=True); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token; tokenizer.padding_side="right"
    key=manifest["model_key"]
    if key in {"qwen7b","phi14b"}:
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
        model=AutoModel.from_pretrained(manifest["model_snapshot"],local_files_only=True,quantization_config=quant,device_map={"":0},dtype=torch.float16).eval(); batch_size=4
    elif key=="qwen32":
        model=AutoModel.from_pretrained(manifest["model_snapshot"],local_files_only=True,dtype=torch.bfloat16,device_map={"":0},attn_implementation="eager",low_cpu_mem_usage=True).eval(); batch_size=1
    else:
        model=AutoModel.from_pretrained(manifest["model_snapshot"],local_files_only=True,dtype=torch.bfloat16,device_map="balanced",max_memory={0:"76GiB",1:"76GiB"},attn_implementation="eager",low_cpu_mem_usage=True).eval(); batch_size=1
    features=[]
    with torch.no_grad():
        for start in range(0,len(selected),batch_size):
            encoded=tokenizer([row["text"] for row in selected[start:start+batch_size]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0")
            hidden=model(**encoded).last_hidden_state; index=encoded["attention_mask"].sum(1)-1
            features.extend(hidden[torch.arange(len(index),device=hidden.device),index].float().cpu())
    output=Path(f"outputs/{key}-{manifest['task']}-final-test/cache/shard-{args.shard:02d}.pt"); output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    tensor=torch.stack(features).half() if features else torch.empty((0,model.config.hidden_size),dtype=torch.float16)
    torch.save({"manifest_sha256":args.manifest_sha256,"shard":args.shard,"indices":list(range(args.shard,args.shard+shards*len(selected),shards)),"features":tensor,"test_evaluated":True},output)

if __name__=="__main__": main()
