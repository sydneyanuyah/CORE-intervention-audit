#!/usr/bin/env python3
"""Four-rank validation-only C3 zero-edit evaluation for F2 XOR readers."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch,torch.distributed as dist,transformers
from torch import nn
from transformers import AutoTokenizer
from train_f1 import load_core,load_json,patch_reader_layers,setup,sha256
class Identity(nn.Module):
 def forward(self,slots,intervention):return slots
def main():
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--graph-seed",type=int,required=True);p.add_argument("--reader",type=Path,required=True);p.add_argument("--expected-reader-sha256",required=True);p.add_argument("--artifact-dir",type=Path,required=True);p.add_argument("--core-components",type=Path,required=True);p.add_argument("--expected-core-sha256",required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if sha256(a.reader)!=a.expected_reader_sha256 or sha256(a.core_components)!=a.expected_core_sha256:raise RuntimeError("C3 XOR frozen hash mismatch")
 ck=torch.load(a.reader,map_location="cpu",weights_only=True)
 if ck.get("seed")!=a.seed or ck.get("graph_seed")!=a.graph_seed or ck.get("test_evaluated") is not False:raise RuntimeError("C3 XOR reader mismatch")
 local,rank=setup("c3",4);device=torch.device("cuda",local);core=load_core(a.core_components);core.seed_all(a.seed);graph=load_json(a.artifact_dir/"graph.json");worlds=load_json(a.artifact_dir/"worlds.json");records=load_json(a.artifact_dir/"records_real.json")
 if graph["seed"]!=a.graph_seed or any(x["split"] not in {"train","validation"} for x in records):raise RuntimeError("C3 XOR data mismatch")
 original=transformers.AutoModel.from_pretrained
 def eager(*x,**kw):kw.setdefault("attn_implementation","eager");return original(*x,**kw)
 transformers.AutoModel.from_pretrained=eager;tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-uncased");reader=core.SharedWorldReader("google-bert/bert-base-uncased",30,split_layer=4,max_world_tokens=16).to(device);patch_reader_layers(reader);reader.load_state_dict(ck["reader"]);reader.eval();cfg=SimpleNamespace(max_length=512,eval_batch_size=8)
 selected,hidden,mask=core.cache_worlds(reader,tokenizer,worlds,"validation",device,cfg);metrics=core.composed_operator_metrics(Identity().to(device),reader,hidden,mask,selected,graph,"world16",cfg)
 if rank==0:
  a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/"run_summary.json").write_text(json.dumps({"protocol":"c3_xor_editor_zeroed_v1","family":"xor","seed":a.seed,"graph_seed":a.graph_seed,"reader_sha256":a.expected_reader_sha256,"world_size":4,"two_edit":metrics,"test_evaluated":False},indent=2,sort_keys=True)+"\n")
 dist.barrier();dist.destroy_process_group();return 0
if __name__=="__main__":raise SystemExit(main())
