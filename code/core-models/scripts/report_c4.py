#!/usr/bin/env python3
"""Measure C4 parameter counts and two-edit improvement per 10k parameters."""

from __future__ import annotations
import hashlib, json, statistics
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from transformers import AutoModel

from core_bert.t3_pointer import T3BPointer
from train_f2_baselines import lora_parameter_count, select_lora_rank

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def tensors(d): return sum(v.numel() for v in d.values() if isinstance(v,torch.Tensor))

def checkpoint_count(path: Path) -> int:
    d=torch.load(path,map_location="cpu",weights_only=False)
    for key in ("operator","state","lora"):
        if isinstance(d.get(key),dict): return tensors(d[key])
    return 0

def metric(d):
    c=d.get("composed",d.get("two_edit",{}))
    for k in ("two_edit_balanced","balanced_numerical_score","balanced_intervention_score"):
        if isinstance(c.get(k),(int,float)): return float(c[k])
    raise RuntimeError("C4 source has no two-edit metric")

def main():
    mp=ROOT/"registry/c4_manifest.json"; m=load(mp); architecture={}
    for label,name in m["model_sizes"].items():
        encoder=AutoModel.from_pretrained(name,attn_implementation="eager")
        hidden=encoder.config.hidden_size; t3=T3BPointer(hidden,m["t3b_rank"])
        complete_t3=sum(p.numel() for p in t3.parameters())
        reader=SimpleNamespace(bert=encoder)
        rank=select_lora_rank(reader,complete_t3); matched=lora_parameter_count(reader,rank)
        architecture[label]={"model":name,"hidden_size":hidden,"t3b_trainable_parameters":complete_t3,"matched_lora_rank":rank,"matched_lora_parameters":matched,"absolute_difference":abs(matched-complete_t3)}
        del encoder,t3
    families={}; cells=[]
    for family in m["f2_families"]:
        rows={method:[] for method in m["f2_methods"]}
        counts={method:[] for method in m["f2_methods"]}
        for seed in m["f2_seeds"]:
            source={}
            for method in m["f2_methods"]:
                candidates=[ROOT/f"outputs/f2-fixed/{family}/{method}/seed-{seed}",ROOT/f"outputs/f2/{family}/{method}/seed-{seed}"]
                folder=next((p for p in candidates if (p/"run_summary.json").is_file()),None)
                if folder is None: raise FileNotFoundError(f"C4 source missing: {family}/{method}/{seed}")
                d=load(folder/"run_summary.json")
                if d.get("method")!=method or d.get("test_evaluated") is not False: raise RuntimeError("C4 source provenance failed")
                count=int(d.get("trainable_parameter_count",checkpoint_count(folder/"operator.pt"))) if method!="prompting" else 0
                rows[method].append(metric(d)); counts[method].append(count); source[method]={"summary_sha256":sha(folder/"run_summary.json"),"parameters":count,"score":rows[method][-1]}
            cells.append({"family":family,"seed":seed,"methods":source})
        baseline=rows["prompting"]; result={}
        for method in m["f2_methods"]:
            mean=statistics.fmean(rows[method]); count=round(statistics.fmean(counts[method]))
            delta=statistics.fmean(a-b for a,b in zip(rows[method],baseline))
            result[method]={"mean_two_edit":mean,"mean_delta_from_prompting":delta,"trainable_parameters":count,"delta_per_10k":None if count==0 else delta*10000/count}
        families[family]=result
    out={"protocol":m["protocol"],"manifest_sha256":sha(mp),"architecture":architecture,"families":families,"cells":cells,"test_evaluated":False}
    (ROOT/"reports/C4_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    lines=["# C4 Parameter accounting","","All counts are measured from instantiated modules or serialized trainable states.","","| Architecture | Complete T3-b params | Matched LoRA rank | Matched LoRA params | Difference |","|---|---:|---:|---:|---:|"]
    for label,d in architecture.items(): lines.append(f"| {label} | {d['t3b_trainable_parameters']:,} | {d['matched_lora_rank']} | {d['matched_lora_parameters']:,} | {d['absolute_difference']:,} |")
    lines += ["","| Family | Method | Two-edit | Params | Delta vs prompting / 10k params |","|---|---|---:|---:|---:|"]
    for family,methods in families.items():
        for method,d in methods.items():
            efficiency = "n/a" if d["delta_per_10k"] is None else f"{d['delta_per_10k']:+.8f}"
            lines.append(f"| {family} | {method} | {d['mean_two_edit']:.6f} | {d['trainable_parameters']:,} | {efficiency} |")
    (ROOT/"reports/C4_REPORT.md").write_text("\n".join(lines)+"\n"); return 0

if __name__=="__main__": raise SystemExit(main())
