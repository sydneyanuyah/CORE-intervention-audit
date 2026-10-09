#!/usr/bin/env python3
"""Four-rank BERT candidate scorer for T5 and G4 baseline/O3 transfer."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,re
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel,AutoTokenizer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def core_load(p):
 s=importlib.util.spec_from_file_location("candidate_core",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def rows(path,split=None):
 out=[]
 for line in Path(path).open():
  r=json.loads(line)
  if r.get("test_evaluated") is not False: raise ValueError("candidate-transfer test isolation failed")
  if split is None or r["split"]==split: out.append(r)
 return out
def letters(options):
 if isinstance(options,list): return list(options)
 found=re.findall(r"(?:^|\n)\s*[-*]?\s*(?:[A-F]|\d+)[\).]\s*(.*?)(?=(?:\n\s*[-*]?\s*(?:[A-F]|\d+)[\).])|$)",options,re.S); return [x.strip() for x in found]
def com2_correct(answer,candidates):
 text=str(answer).strip(); labels=re.findall(r"(?:^|,|\band\b)\s*([A-F]|\d+)\s*(?=[,\.\)]|\band\b|$)",text)
 if not labels:
  lead=re.match(r"\s*([A-F]|\d+)\s*(?:[\).:]|\()",text); labels=[lead.group(1)] if lead else []
 if labels:
  indices=[ord(x)-65 if x in "ABCDEF" else int(x)-1 for x in labels]
  if any(i<0 or i>=len(candidates) for i in indices):raise ValueError("Com2 answer index out of range")
  return {candidates[i] for i in indices}
 normalized=text.rstrip(".").strip().lower(); matches={x for x in candidates if x.rstrip(".").strip().lower()==normalized}
 if not matches:raise ValueError(f"unresolved Com2 answer: {answer}")
 return matches
def expand(records,task):
 result=[]
 for r in records:
  if task=="g4": prompt=f"Intervention: {r['intervention']}. Comparator: {r['comparator']}. Outcome: {r['outcome']}."; candidates=["decreased","no_change","increased"]; correct={r["label"]}; category="evidence_inference"
  elif task=="g1": prompt=f"{r['given_info']} Question: {r['question']}"; candidates=["ate","backadj","collider_bias","correlation","det-counterfactual","ett","exp_away","marginal","nde","nie"]; correct={r["query_type"]}; category="estimand_selection"
  elif r["source"]=="crass":
   prompt=f"{r['premise']} {r['question']}"; candidates=letters(r["options"]); candidates += [] if r["answer"] in candidates else [r["answer"]]; correct={r["answer"]}; category="crass"
  elif r["source"]=="counterbench": prompt=f"{r['context']} {r['question']}"; candidates=["no","yes"]; correct={str(r["answer"]).lower()}; category="counterbench"
  elif r["source"]=="corr2cause": prompt=r["input"]; candidates=["0","1"]; correct={str(r["answer"])}; category="corr2cause"
  elif r["source"]=="com2":
   prompt=f"{r['scenario']} {r['question']}"; candidates=letters(r["options"]); correct=com2_correct(r["answer"],candidates); category="com2"
  else: raise ValueError("unknown native transfer source")
  if not candidates: raise ValueError(f"no candidates: {r['id']}")
  for i,candidate in enumerate(candidates):
   label=int(candidate in correct); key=f"{category}:{i}"
   if task=="g1": command=i
   elif task=="t5": command={"crass":0,"counterbench":10,"corr2cause":20,"com2":30}[category]+i
   else: command=int.from_bytes(hashlib.sha256(key.encode()).digest()[:2],"big")%256
   result.append({"id":r["id"],"group":r.get("article_group_id",r.get("group_id",r["id"])),"source":r.get("source","g4"),"text":f"{prompt} Candidate answer: {candidate}","label":label,"command":command})
 return result
class Model(nn.Module):
 def __init__(self,core,method,operator_state=None):
  super().__init__(); self.bert=AutoModel.from_pretrained("google-bert/bert-base-uncased",attn_implementation="eager",add_pooling_layer=False); self.method=method; h=self.bert.config.hidden_size; self.state=nn.Parameter(torch.randn(16,h)*.02); self.operator=core.O3StateGated(60 if operator_state is not None else 256,h,16); self.head=nn.Linear(h,1)
  if operator_state is not None:
   self.operator.load_state_dict(operator_state)
   for parameter in self.operator.parameters(): parameter.requires_grad_(False)
   for parameter in self.bert.parameters(): parameter.requires_grad_(False)
 def bias(self,m,d):return (1-m[:,None,None,:].to(d))*torch.finfo(d).min
 def layers(self,x,m,lo,hi):
  b=self.bias(m,x.dtype)
  for layer in self.bert.encoder.layer[lo:hi]:
   output=layer(x,attention_mask=b); x=output[0] if isinstance(output,(tuple,list)) else output
  return x
 def forward(self,ids,mask,cmd):
  x=self.bert.embeddings(input_ids=ids); state=self.state[None].expand(len(x),-1,-1); x=torch.cat([x,state],1); mask=torch.cat([mask,torch.ones(len(x),16,dtype=mask.dtype,device=mask.device)],1); x=self.layers(x,mask,0,4)
  if self.method=="o3": x=x.clone(); x[:,-16:]=self.operator(x[:,-16:],cmd)
  x=self.layers(x,mask,4,12); return self.head(x[:,0]).squeeze(-1)
def encoded_batch(enc,indices):
 if not enc.get("dynamic"):
  return enc["input_ids"][indices],enc["attention_mask"][indices],enc["command"][indices]
 positions=indices.tolist(); sequences=[enc["input_ids"][i] for i in positions]; width=max(len(x) for x in sequences)
 ids=torch.full((len(sequences),width),enc["pad_token_id"],dtype=torch.long); mask=torch.zeros((len(sequences),width),dtype=torch.long)
 for row,sequence in enumerate(sequences):
  ids[row,:len(sequence)]=torch.tensor(sequence,dtype=torch.long);mask[row,:len(sequence)]=1
 return ids,mask,torch.tensor([enc["command"][i] for i in positions],dtype=torch.long)
def evaluate(model,enc,data,rank,device):
 local=[]; model.eval()
 with torch.no_grad():
  idx=torch.arange(len(data))[rank::4]
  for start in range(0,len(idx),16):
   sl=idx[start:start+16];ids,mask,command=encoded_batch(enc,sl);score=model(ids.to(device),mask.to(device),command.to(device)).cpu()
   local.extend((data[i]["id"],data[i]["group"],data[i]["source"],data[i]["label"],float(s)) for i,s in zip(sl.tolist(),score))
 gathered=[None]*4; dist.all_gather_object(gathered,local); flat=[x for part in gathered for x in part]; by_id=defaultdict(list)
 for row in flat:by_id[row[0]].append(row)
 correct=0; by_source=defaultdict(lambda:[0,0]); by_article=defaultdict(list)
 for group in by_id.values():
  pred=max(range(len(group)),key=lambda i:group[i][4]); gold=[i for i,x in enumerate(group) if x[3]]; hit=int(pred in gold); correct+=hit; by_source[group[0][2]][0]+=hit; by_source[group[0][2]][1]+=1; by_article[group[0][1]].append(hit)
 article_accuracy=sum(sum(v)/len(v) for v in by_article.values())/len(by_article)
 return {"accuracy":article_accuracy,"prompt_accuracy":correct/len(by_id),"groups":len(by_id),"article_groups":len(by_article),"aggregation":"macro mean of prompt accuracy within article/group","by_source":{k:{"accuracy":v[0]/v[1],"groups":v[1]} for k,v in sorted(by_source.items())}}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--task",choices=("t5","g1","g4"),required=True); p.add_argument("--method",choices=("baseline","o3"),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--manifest-sha256",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--token-cache",type=Path); a=p.parse_args(); m=json.loads(a.manifest.read_text())
 if sha(a.manifest)!=a.manifest_sha256 or sha(m["operator_source"])!=m["operator_source_sha256"]:raise ValueError("candidate-transfer identity mismatch")
 if a.task in {"t5","g1"}:
  if sha(m["train_data"])!=m["train_data_sha256"] or sha(m["validation_data"])!=m["validation_data_sha256"]:raise ValueError("T5 split-file identity mismatch")
  train_rows=rows(m["train_data"]); val_rows=rows(m["validation_data"])
  if any(r.get("split")!="train" for r in train_rows) or any(r.get("split")!="validation" for r in val_rows):raise ValueError("physical split isolation failed")
 else:
  if sha(m["data"])!=m["data_sha256"]:raise ValueError("G4 data identity mismatch")
  train_rows=rows(m["data"],"train"); val_rows=rows(m["data"],"validation")
 cells=[c for c in m.get("cells",[]) if c["seed"]==a.seed and c["method"]==a.method]
 if a.seed not in m["seeds"] or a.method not in m["methods"] or int(os.environ.get("WORLD_SIZE","1"))!=4 or (m.get("cells") and len(cells)!=1):raise ValueError("unregistered candidate-transfer cell")
 dist.init_process_group("nccl"); rank=dist.get_rank(); local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); device=torch.device("cuda",local); torch.manual_seed(a.seed)
 train=expand(train_rows,a.task); val=expand(val_rows,a.task)
 if a.task=="g1" and a.token_cache and a.token_cache.is_file():
  cached=torch.load(a.token_cache,map_location="cpu",weights_only=False)
  expected={"version":1,"model":m["model"],"train_data_sha256":m["train_data_sha256"],"validation_data_sha256":m["validation_data_sha256"]}
  if any(cached.get(k)!=v for k,v in expected.items()) or len(cached.get("train",{}).get("input_ids",[]))!=len(train) or len(cached.get("validation",{}).get("input_ids",[]))!=len(val):raise ValueError("G1 token-cache identity mismatch")
  tr=cached["train"];va=cached["validation"]
 else:
  tok=AutoTokenizer.from_pretrained(m["model"]); tr=tok([x["text"] for x in train],padding=True,truncation=True,max_length=512,return_tensors="pt"); va=tok([x["text"] for x in val],padding=True,truncation=True,max_length=512,return_tensors="pt"); tr["command"]=torch.tensor([x["command"] for x in train]); va["command"]=torch.tensor([x["command"] for x in val])
 y=torch.tensor([x["label"] for x in train],dtype=torch.float)
 operator_state=None
 if cells:
  checkpoint=Path(cells[0]["checkpoint"])
  if sha(checkpoint)!=cells[0]["checkpoint_sha256"]:raise ValueError("frozen F1 operator identity mismatch")
  payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
  if payload.get("method")!="o3" or payload.get("test_evaluated") is not False:raise ValueError("invalid frozen F1 O3 checkpoint")
  operator_state=payload["operator"]
 model=DDP(Model(core_load(Path(m["operator_source"])),a.method,operator_state).to(device),device_ids=[local],broadcast_buffers=False,find_unused_parameters=a.method=="baseline"); opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=3e-5); best=-1; selected=None; state=None
 for epoch in range(3):
  order=torch.randperm(len(train),generator=torch.Generator().manual_seed(a.seed+epoch))[rank::4]; model.train()
  for start in range(0,len(order),8):
   idx=order[start:start+8];ids,mask,command=encoded_batch(tr,idx);logits=model(ids.to(device),mask.to(device),command.to(device)); loss=F.binary_cross_entropy_with_logits(logits,y[idx].to(device)); opt.zero_grad(); loss.backward(); opt.step()
  metrics=evaluate(model,va,val,rank,device)
  if rank==0 and metrics["accuracy"]>best:
   best=metrics["accuracy"];selected=epoch+1
   state={k:v.detach().cpu().clone() for k,v in model.module.state_dict().items() if operator_state is None or k=="state" or k.startswith("head.")}
   best_metrics=metrics
 if rank==0:
        a.output.mkdir(parents=True,exist_ok=False); torch.save({"model":state,"task":a.task,"method":a.method,"seed":a.seed,"frozen_operator_checkpoint_sha256":cells[0]["checkpoint_sha256"] if cells else None,"test_evaluated":False},a.output/"best.pt"); (a.output/"run_summary.json").write_text(json.dumps({"protocol":m["protocol"],"task":a.task,"method":a.method,"seed":a.seed,"selected_epoch":selected,"validation":best_metrics,"world_size":4,"manifest_sha256":a.manifest_sha256,"frozen_operator_checkpoint_sha256":cells[0]["checkpoint_sha256"] if cells else None,"test_evaluated":False},indent=2,sort_keys=True)+"\n")
 dist.barrier();dist.destroy_process_group();return 0
if __name__=="__main__":raise SystemExit(main())
