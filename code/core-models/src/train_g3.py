#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import train_t2
def main():
 p=argparse.ArgumentParser();p.add_argument("--layer",type=int,required=True);known,rest=p.parse_known_args();layer=known.layer
 original_read=train_t2.read_seed;original_model=train_t2.T2Model
 train_t2.read_seed=lambda path,unused_seed:original_read(path,501)
 class LayerModel(original_model):
  def __init__(self,core,model_name,unused_layer,world_tokens,max_nodes):super().__init__(core,model_name,layer,world_tokens,max_nodes)
 train_t2.T2Model=LayerModel;sys.argv=[sys.argv[0],*rest];code=train_t2.main()
 if int(os.environ.get("LOCAL_RANK","0"))==0:
  out=Path(rest[rest.index("--output")+1])/"run_summary.json";d=json.loads(out.read_text());d["candidate_layer"]=layer;d["data_generation_seed"]=501;out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
 return code
if __name__=="__main__":raise SystemExit(main())
