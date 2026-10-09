#!/usr/bin/env python3
"""Four-rank, validation-only C3 editor-zeroed measurement for repaired Com2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch,torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer
from core_bert.com2_two_edit_adapter import load_com2_two_edit_bundle
from core_bert.composed_reader import ComposedOpenTextModel,ComposedScientificCollator,ComposedScientificExecutor
from core_bert.open_text_outputs import normalize_open_text
from core_bert.two_edit import two_edit_balanced_metrics
from evaluate_xor_two_edit import _load_model
from train_f1 import setup
from train_f2_com2 import move,sha256,validate_contract

class UnusedEditor(nn.Module):
 def forward(self,slots,instruction): return slots

def main():
 p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,required=True); p.add_argument("--f2-operator",type=Path,required=True); p.add_argument("--expected-f2-operator-sha256",required=True)
 for name in ("checkpoint","manifest","cells","source-catalog","amendment","queue","truth","provenance"):
  p.add_argument(f"--{name}",type=Path,required=True); p.add_argument(f"--expected-{name.replace('_','-')}-sha256",required=True)
 p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
 for name in ("intervention-records","counterfactual-records","intervention-train","counterfactual-train","intervention-validation","counterfactual-validation","groups"): p.add_argument(f"--expected-{name}-sha256",required=True)
 a=p.parse_args(); a.method="o3"; source=validate_contract(a)
 if sha256(a.f2_operator)!=a.expected_f2_operator_sha256: raise RuntimeError("C3 F2 operator/head hash mismatch")
 local,rank=setup("c3",4); device=torch.device("cuda",local); tokenizer=AutoTokenizer.from_pretrained(a.tokenizer)
 checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False); loaded=_load_model(SimpleNamespace(family="com2",mode="t2b",a2_control="active",model=None),checkpoint,tokenizer,device)
 f2=torch.load(a.f2_operator,map_location="cpu",weights_only=False); loaded.open_text_head.load_state_dict(f2["open_text_head"]); loaded.eval()
 bundle=load_com2_two_edit_bundle(a.queue,a.truth,a.provenance,a.data_root,split="validation"); mine=bundle.examples[rank::4]; predictions={}
 executor=ComposedScientificExecutor(loaded.executor.reader,UnusedEditor().to(device),"t2b",editor_enabled=False)
 model=ComposedOpenTextModel(executor,loaded.open_text_head,max_tokens=32)
 with torch.no_grad():
  for start in range(0,len(mine),8):
   examples=mine[start:start+8]; batch=ComposedScientificCollator(tokenizer,bundle.records,max_length=512)(examples); batch=type(batch)(batch.examples,move(batch.first,device),move(batch.second,device)); out=model(batch); text=tokenizer.batch_decode(out.generated_token_ids.cpu(),skip_special_tokens=True); cursor=0
   for ex in examples:
    width=len(ex.gold_outputs); predictions[ex.pair_id]=[normalize_open_text(x) for x in text[cursor:cursor+width]]; cursor+=width
 gathered=[None]*4; dist.all_gather_object(gathered,predictions)
 if rank==0:
  merged={}; [merged.update(x) for x in gathered]; metrics=two_edit_balanced_metrics(bundle.examples,merged); a.output_dir.mkdir(parents=True,exist_ok=False)
  row={"protocol":"c3_com2_editor_zeroed_v1","family":"com2","seed":a.seed,"source_a1_cell_id":source["cell_id"],"checkpoint_sha256":sha256(a.checkpoint),"f2_operator_sha256":sha256(a.f2_operator),"world_size":4,"two_edit":metrics,"test_evaluated":False}; (a.output_dir/"run_summary.json").write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
 dist.barrier();dist.destroy_process_group();return 0
if __name__=="__main__": raise SystemExit(main())
