#!/usr/bin/env python3
"""Create the immutable execution ledger for the 23-family Phi-4 14B expansion."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
rows=[
 (1,"F1",[],40),(2,"F2",["F1"],40),(3,"F3",["F1"],80),(4,"F4",["F1"],50),
 (5,"A1",[],120),(6,"A2",["A1"],20),(7,"A3",["A1"],20),
 (8,"L1",[],120),(9,"L2",["L1"],100),(10,"L3",["F1"],40),
 (11,"C1",["F1"],60),(12,"C2",["F1"],None),(13,"C3",["F2"],None),(14,"C4",["F2"],None),
 (15,"T1",["F2"],None),(16,"T2",[],40),(17,"T3",["A1"],40),(18,"T4",[],40),(19,"T5",["F1"],40),
 (20,"G1",["F1"],40),(21,"G2",["F1"],40),(22,"G3",[],8),(23,"G4",[],40),
]
q={"protocol":"core_phi4_14b_23_experiment_queue_v1","model":"microsoft/phi-4","model_revision":"not-published","policy":{"fill_all_genuinely_free_gpus":True,"launch_only_registered_provenance_valid_cells":True,"never_rerun_complete_cells":True,"held_out_test":"forbidden","orchestration":"persistent server-side pipeline; no Codex scheduled task"},"experiments":[{"number":n,"id":x,"depends_on":d,"state":"queued","expected_cells":c} for n,x,d,c in rows],"execution_order":[x for _,x,_,_ in rows],"queue_owner":"scripts/run_phi14b_all.py","test_evaluated":False}
(R/"registry/phi14b_23_queue.json").write_text(json.dumps(q,indent=2,sort_keys=True)+"\n")
