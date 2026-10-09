#!/usr/bin/env python3
"""Four-rank BERT-base/O3 T2 training on CSuite development folds."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("t2_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_seed(path: Path, seed: int) -> list[dict]:
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("test_evaluated") is not False or row.get("split") != "development_generated":
            raise ValueError(f"forbidden T2 row in {path}")
        if int(row["generation_seed"]) == seed:
            rows.append(row)
    return rows


def direction(after: float, before: float, tolerance: float = 1e-7) -> int:
    delta = float(after) - float(before)
    return 2 if delta > tolerance else 0 if delta < -tolerance else 1


def prepare(rows: list[dict], max_nodes: int):
    texts, labels, masks, changed, intervention_ids, families = [], [], [], [], [], []
    for row in rows:
        factual, target = row["factual_state"], row["intervened_state"]
        nodes = sorted(target, key=lambda x: int(x[1:]))
        if len(nodes) > max_nodes or nodes != [f"x{i}" for i in range(len(nodes))]:
            raise ValueError(f"invalid T2 node layout: {row['id']}")
        prefix = row["rendered_input"].split(" Apply do(", 1)[0]
        values = [direction(target[node], factual[node]) for node in nodes]
        labels.append(values + [1] * (max_nodes - len(nodes)))
        masks.append([True] * len(nodes) + [False] * (max_nodes - len(nodes)))
        changed.append([value != 1 for value in values] + [False] * (max_nodes - len(nodes)))
        intervention = row["intervention"]
        index = int(intervention["target"][1:])
        command_direction = direction(intervention["value"], intervention["reference_value"])
        intervention_ids.append(index * 3 + command_direction)
        texts.append(prefix); families.append(row["sem_family"])
    return texts, torch.tensor(labels), torch.tensor(masks), torch.tensor(changed), torch.tensor(intervention_ids), families


class T2Model(nn.Module):
    def __init__(self, core, model_name: str, split_layer: int, world_tokens: int, max_nodes: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="eager", add_pooling_layer=False)
        self.split_layer = split_layer; self.world_tokens = world_tokens; self.max_nodes = max_nodes
        hidden = self.bert.config.hidden_size
        self.state = nn.Parameter(torch.randn(world_tokens, hidden) * 0.02)
        self.queries = nn.Parameter(torch.randn(max_nodes, hidden) * 0.02)
        self.operator = core.O3StateGated(max_nodes * 3, hidden, 16)
        self.head = nn.Linear(hidden, 3)

    @staticmethod
    def bias(mask, dtype):
        return (1.0 - mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min

    def run_layers(self, layers, hidden, mask):
        bias = self.bias(mask, hidden.dtype)
        for layer in layers:
            output = layer(hidden, attention_mask=bias)
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        return hidden

    def forward(self, input_ids, attention_mask, intervention_id, return_law_state=False):
        hidden = self.bert.embeddings(input_ids=input_ids)
        state = self.state[None].expand(len(hidden), -1, -1)
        hidden = torch.cat([hidden, state], 1)
        mask = torch.cat([attention_mask, torch.ones(len(hidden), self.world_tokens, dtype=attention_mask.dtype, device=attention_mask.device)], 1)
        hidden = self.run_layers(self.bert.encoder.layer[:self.split_layer], hidden, mask)
        pre_state = hidden[:, -self.world_tokens:]
        hidden = hidden.clone(); hidden[:, -self.world_tokens:] = self.operator(pre_state, intervention_id)
        queries = self.queries[None].expand(len(hidden), -1, -1)
        hidden = torch.cat([hidden, queries], 1)
        mask = torch.cat([mask, torch.ones(len(hidden), self.max_nodes, dtype=mask.dtype, device=mask.device)], 1)
        hidden = self.run_layers(self.bert.encoder.layer[self.split_layer:], hidden, mask)
        logits = self.head(hidden[:, -self.max_nodes:])
        if return_law_state:
            return logits, law_loss(self.operator, pre_state, intervention_id, self.max_nodes)
        return logits


def law_loss(operator, hidden, ids, max_nodes: int):
    noop = torch.full_like(ids, max_nodes * 3)
    identity = F.mse_loss(operator(hidden, noop), hidden)
    once = operator(hidden, ids)
    idempotence = F.mse_loss(operator(once, ids), once)
    variable, value = ids // 3, ids % 3
    other = ((variable + 7) % max_nodes) * 3 + ((value + 1) % 3)
    commutation = F.mse_loss(operator(operator(hidden, ids), other), operator(operator(hidden, other), ids))
    prior = variable * 3 + ((value + 1) % 3)
    last_write = F.mse_loss(operator(operator(hidden, prior), ids), operator(hidden, ids))
    return identity + idempotence + commutation + last_write


def evaluate(model, encoded, labels, masks, changed, families, rank, world_size, device):
    model.eval(); counts = torch.zeros(6, dtype=torch.float64, device=device); by_family = defaultdict(lambda: torch.zeros(6, dtype=torch.float64, device=device))
    with torch.no_grad():
        indices = torch.arange(len(labels))[rank::world_size]
        for start in range(0, len(indices), 16):
            idx = indices[start:start + 16]
            logits = model(encoded["input_ids"][idx].to(device), encoded["attention_mask"][idx].to(device), encoded["intervention_id"][idx].to(device))
            pred = logits.argmax(-1).cpu(); y=labels[idx]; valid=masks[idx]; ch=changed[idx]; same=valid & ~ch
            batch = torch.tensor([(pred[valid]==y[valid]).sum().item(),valid.sum().item(),(pred[ch]==y[ch]).sum().item(),ch.sum().item(),(pred[same]==y[same]).sum().item(),same.sum().item()],dtype=torch.float64,device=device)
            counts += batch
            for local,row_index in enumerate(idx.tolist()):
                v=valid[local]; c=ch[local]; s=same[local]; p=pred[local]; t=y[local]
                by_family[families[row_index]] += torch.tensor([(p[v]==t[v]).sum().item(),v.sum().item(),(p[c]==t[c]).sum().item(),c.sum().item(),(p[s]==t[s]).sum().item(),s.sum().item()],dtype=torch.float64,device=device)
    dist.all_reduce(counts)
    gathered=[None for _ in range(world_size)]; dist.all_gather_object(gathered,{k:v.cpu().tolist() for k,v in by_family.items()})
    merged=defaultdict(lambda:[0.0]*6)
    for item in gathered:
        for family,vals in item.items(): merged[family]=[a+b for a,b in zip(merged[family],vals)]
    def metrics(c):
        acc=c[0]/max(c[1],1); change=c[2]/max(c[3],1); preserve=c[4]/max(c[5],1)
        return {"accuracy":acc,"change_accuracy":change,"preservation_accuracy":preserve,"balanced_direction_accuracy":0.5*(change+preserve),"cells":int(c[1])}
    return metrics(counts.tolist()),{k:metrics(v) for k,v in sorted(merged.items())}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--manifest-sha256",required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if sha256(args.manifest)!=args.manifest_sha256 or args.seed not in manifest["seeds"]: raise ValueError("T2 manifest identity mismatch")
    train_path=Path(manifest["train_data"]); val_path=Path(manifest["validation_data"]); core_path=Path(manifest["operator_source"])
    for path,key in [(train_path,"train_sha256"),(val_path,"validation_sha256"),(core_path,"operator_source_sha256")]:
        if sha256(path)!=manifest[key]: raise ValueError(f"T2 input hash mismatch: {path}")
    if int(os.environ.get("WORLD_SIZE","1"))!=4: raise RuntimeError("T2 requires exactly four ranks")
    dist.init_process_group("nccl"); rank=dist.get_rank(); local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); device=torch.device("cuda",local)
    torch.manual_seed(args.seed); core=load_core(core_path); train=read_seed(train_path,args.seed); val=read_seed(val_path,args.seed)
    train_parts=prepare(train,manifest["max_nodes"]); val_parts=prepare(val,manifest["max_nodes"])
    tokenizer=AutoTokenizer.from_pretrained(manifest["model"])
    tr_enc=tokenizer(train_parts[0],padding=True,truncation=True,max_length=512,return_tensors="pt"); va_enc=tokenizer(val_parts[0],padding=True,truncation=True,max_length=512,return_tensors="pt")
    tr_enc["intervention_id"]=train_parts[4]; va_enc["intervention_id"]=val_parts[4]
    base=T2Model(core,manifest["model"],manifest["split_layer"],manifest["world_tokens"],manifest["max_nodes"]).to(device)
    with torch.no_grad():
        ids=[tokenizer(f"x{i}",add_special_tokens=False)["input_ids"] for i in range(manifest["max_nodes"])]
        base.queries.copy_(torch.stack([base.bert.embeddings.word_embeddings(torch.tensor(x,device=device)).mean(0) for x in ids]))
    model=DDP(base,device_ids=[local],broadcast_buffers=False); optimizer=torch.optim.AdamW(model.parameters(),lr=manifest["learning_rate"],weight_decay=0.01); history=[]
    best_score=-math.inf; best_state=None; selected_epoch=None; selected_overall=None; selected_by_family=None
    labels,masks,changed=train_parts[1:4]
    for epoch in range(manifest["epochs"]):
        order=torch.randperm(len(labels),generator=torch.Generator().manual_seed(args.seed+epoch))[rank::4]; model.train(); losses=[]
        for start in range(0,len(order),manifest["batch_size_per_rank"]):
            idx=order[start:start+manifest["batch_size_per_rank"]]; ids=tr_enc["input_ids"][idx].to(device); attn=tr_enc["attention_mask"][idx].to(device); commands=tr_enc["intervention_id"][idx].to(device); y=labels[idx].to(device); valid=masks[idx].to(device); ch=changed[idx].to(device); same=valid & ~ch
            logits,laws=model(ids,attn,commands,True); per=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction="none").reshape_as(y)
            task=0.5*((per[ch].mean() if ch.any() else per[valid].mean())+(per[same].mean() if same.any() else per[valid].mean())); loss=task+manifest["law_weight"]*laws
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        overall,by_family=evaluate(model,va_enc,val_parts[1],val_parts[2],val_parts[3],val_parts[5],rank,4,device); history.append({"epoch":epoch+1,"loss":statistics_fmean(losses),"validation":overall})
        if rank==0 and overall["balanced_direction_accuracy"]>best_score:
            best_score=overall["balanced_direction_accuracy"]; selected_epoch=epoch+1; selected_overall=overall; selected_by_family=by_family
            best_state={key:value.detach().cpu().clone() for key,value in model.module.state_dict().items()}
    if rank==0:
        if best_state is None or selected_overall is None or selected_by_family is None: raise RuntimeError("T2 validation selection failed")
        args.output.mkdir(parents=True,exist_ok=False); torch.save({"model":best_state,"seed":args.seed,"selected_epoch":selected_epoch,"manifest_sha256":args.manifest_sha256,"test_evaluated":False},args.output/"best.pt")
        summary={"protocol":manifest["protocol"],"seed":args.seed,"world_size":4,"train_rows":len(train),"validation_rows":len(val),"selected_epoch":selected_epoch,"selection_metric":"validation_balanced_direction_accuracy","validation":selected_overall,"by_sem_family":selected_by_family,"history":history,"manifest_sha256":args.manifest_sha256,"train_sha256":manifest["train_sha256"],"validation_sha256":manifest["validation_sha256"],"test_evaluated":False}
        (args.output/"run_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    dist.barrier(); dist.destroy_process_group(); return 0


def statistics_fmean(values):
    return sum(values)/len(values) if values else math.nan


if __name__=="__main__": raise SystemExit(main())
