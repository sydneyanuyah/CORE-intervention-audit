#!/usr/bin/env python3
"""T4 arm-filtered CSuite training through the provenance-tested T2 DDP core."""
from __future__ import annotations
import argparse,json,os,re,sys
from pathlib import Path
import train_t2

def main():
 p=argparse.ArgumentParser(); p.add_argument("--arm",required=True); known,rest=p.parse_known_args(); arm=known.arm
 original_read=train_t2.read_seed; original_prepare=train_t2.prepare
 def read(path,seed):
  rows=original_read(path,seed); selected=[r for r in rows if r.get("arm")==arm]
  if not selected: raise ValueError(f"no T4 rows for {arm}/{seed}")
  return selected
 def prepare(rows,max_nodes):
  converted=[]
  for row in rows:
   match=re.search(r"do\((x\d+)\s*=\s*(-?\d+(?:\.\d+)?)\)",row["rendered_input"])
   if match is None: match=re.search(r"Change (x\d+) from (-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?)",row["rendered_input"])
   if match is None: raise ValueError(f"cannot parse T4 intervention: {row['id']}")
   if len(match.groups())==2: target,new_value=match.groups(); reference=row["factual_state"][target]
   else: target,reference,new_value=match.groups()
   converted_row=dict(row); converted_row["intervention"]={"target":target,"value":float(new_value),"reference_value":float(reference)}; converted.append(converted_row)
  return original_prepare(converted,max_nodes)
 train_t2.read_seed=read; train_t2.prepare=prepare
 sys.argv=[sys.argv[0],*rest]; code=train_t2.main()
 if int(os.environ.get("LOCAL_RANK","0"))==0:
  out=Path(rest[rest.index("--output")+1])/"run_summary.json"; d=json.loads(out.read_text()); d["arm"]=arm; out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
 return code
if __name__=="__main__": raise SystemExit(main())
