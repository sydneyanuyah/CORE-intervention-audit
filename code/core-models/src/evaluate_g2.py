#!/usr/bin/env python3
"""Frozen-operator G2 shared-target last-write-wins measurement."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os
from pathlib import Path
import torch
import torch.distributed as dist

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load_core(path: Path):
    spec=importlib.util.spec_from_file_location("g2_core",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def base_state(rows, device):
    # A fixed, nonlearned algebraic probe state. G2 measures the operator law,
    # not downstream task accuracy, so no reader/checkpoint is trained here.
    dim=torch.arange(768,device=device,dtype=torch.float32)[None,None,:]+1
    slots=torch.arange(16,device=device,dtype=torch.float32)[None,:,None]+1
    state=torch.sin(slots*dim*0.0017).expand(len(rows),-1,-1).clone()
    for i,row in enumerate(rows):
        for name,value in row["factual_state"].items(): state[i,int(name[1:])] += (2*int(value)-1)*0.05
    return state
def main():
    p=argparse.ArgumentParser(); p.add_argument("--method",choices=("o2","o3"),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--manifest-sha256",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    m=json.loads(a.manifest.read_text()); cells=[c for c in m["cells"] if c["method"]==a.method and c["seed"]==a.seed]
    if sha(a.manifest)!=a.manifest_sha256 or len(cells)!=1 or int(os.environ.get("WORLD_SIZE","1"))!=4: raise ValueError("unregistered G2 cell")
    c=cells[0]; data=Path(m["validation_data"]); core_path=Path(m["operator_source"]); checkpoint=Path(c["checkpoint"])
    if sha(data)!=m["validation_data_sha256"] or sha(core_path)!=m["operator_source_sha256"] or sha(checkpoint)!=c["checkpoint_sha256"]: raise ValueError("G2 frozen identity mismatch")
    rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    if not rows or any(r.get("split")!="validation" or r.get("test_evaluated") is not False or r.get("law")!="last_write_wins" for r in rows): raise ValueError("G2 validation/test contract failed")
    dist.init_process_group("nccl"); rank=dist.get_rank(); local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); device=torch.device("cuda",local)
    core=load_core(core_path)
    operator=(core.O2ConditionalLowRank(60,16,768,16) if a.method=="o2" else core.O3StateGated(60,768,16)).to(device)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False); operator.load_state_dict(payload["operator"]); operator.eval()
    selected=rows[rank::4]; sums=torch.zeros(3,device=device,dtype=torch.float64)
    with torch.no_grad():
        for start in range(0,len(selected),32):
            batch=selected[start:start+32]; base=base_state(batch,device); full=base
            for step in range(max(len(r["ordered_edits"]) for r in batch)):
                active=[i for i,r in enumerate(batch) if step<len(r["ordered_edits"])]
                if not active: continue
                ids=torch.tensor([int(batch[i]["ordered_edits"][step]["target"][1:])*2+int(batch[i]["ordered_edits"][step]["value"]) for i in active],device=device)
                full=full.clone(); full[active]=operator(full[active],ids)
            last_ids=torch.tensor([int(r["ordered_edits"][-1]["target"][1:])*2+int(r["ordered_edits"][-1]["value"]) for r in batch],device=device)
            last=operator(base,last_ids); gap=(full-last).flatten(1).norm(dim=1)/last.flatten(1).norm(dim=1).clamp_min(1e-12)
            sums += torch.tensor([gap.sum().item(),(gap<=1e-5).sum().item(),len(batch)],device=device,dtype=torch.float64)
    dist.all_reduce(sums); result={"mean_normalized_hidden_gap":float(sums[0]/sums[2]),"hidden_agreement_at_1e-5":float(sums[1]/sums[2]),"sequences":int(sums[2])}
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=False); (a.output/"measurement_summary.json").write_text(json.dumps({"protocol":m["protocol"],"method":a.method,"seed":a.seed,"checkpoint_sha256":c["checkpoint_sha256"],"validation":result,"world_size":4,"manifest_sha256":a.manifest_sha256,"test_evaluated":False},indent=2,sort_keys=True)+"\n")
    dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
