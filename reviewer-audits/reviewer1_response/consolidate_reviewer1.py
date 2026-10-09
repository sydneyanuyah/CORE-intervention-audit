#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path

base=Path(sys.argv[1]); raw=base/"raw"
rows=[]; macro=[]
d=json.loads((raw/"bert_selective_invariance.json").read_text())
for r in d["per_seed"]:
    rows.append({"backbone":r["backbone"],"experiment":r["experiment"],"family":r["family"],"method":r["method"],"seed":r["seed"],"cell_id":r["cell_id"],"non_descendant_preservation":r["non_descendant_preservation"],"descendant_change_sensitivity":r["descendant_change_sensitivity"],"balanced_selective_invariance":r["selective_invariance_balanced"],"checkpoint_sha256":r["checkpoint_sha256"],"metric_source_path":r["summary_path"],"metric_source_sha256":r["summary_sha256"],"split":"validation","test_evaluated":False})
macro += d["aggregate"]
for p in sorted(raw.glob("*_selective_invariance.json")):
    if p.name.startswith("bert_"): continue
    d=json.loads(p.read_text())
    for r in d["per_seed"]:
        rows.append({"backbone":r["backbone"],"experiment":r["experiment"],"family":"cladder","method":r["method"],"seed":r["seed"],"cell_id":r["cell_id"],"non_descendant_preservation":r["non_descendant_preservation"],"descendant_change_sensitivity":r["descendant_change_sensitivity"],"balanced_selective_invariance":r["balanced_selective_invariance"],"checkpoint_sha256":r["checkpoint_sha256"],"metric_source_path":r["metric_source_path"],"metric_source_sha256":r["metric_source_sha256"],"split":r["evaluation_split"],"test_evaluated":r["test_evaluated"]})
    macro += d["macro"]
with (base/"CORE_REVIEWER1_SELECTIVE_INVARIANCE_PER_SEED.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(base/"CORE_REVIEWER1_SELECTIVE_INVARIANCE.json").write_text(json.dumps({"protocol":"reviewer1_selective_invariance_consolidated_v1","row_count":len(rows),"backbones":sorted({r['backbone'] for r in rows}),"per_seed":rows,"macro":macro},indent=2)+"\n")
f2=[]
d=json.loads((raw/"bert_f2_uncertainty.json").read_text())
for x in d["families"]:
    f2.append({"backbone":x["backbone"],"family":x["family"],"n_seeds":x["n_seeds"],"n_graphs":x["n_graphs"],"mean_o2_minus_o3":x["paired_seed_mean"],"student_t_95":x["paired_seed_student_t_95"],"equal_graph_mean_o2_minus_o3":x["equal_graph_mean"],"graph_bootstrap_95":x["paired_graph_bootstrap_95"],"eligible_for_paper":x["family"]=="cladder","source_file":x})
for p in sorted(raw.glob("*_f2_ci.json")):
    d=json.loads(p.read_text());f2.append({"backbone":d["backbone"],"family":"cladder","n_seeds":d["paired_seed_count"],"n_graphs":d["graph_count"],"mean_o2_minus_o3":d["mean_o2_minus_o3"],"student_t_95":d["paired_seed_student_t_95"],"equal_graph_mean_o2_minus_o3":d["equal_graph_mean_o2_minus_o3"],"graph_bootstrap_95":d["paired_graph_bootstrap_95"],"eligible_for_paper":True,"source_file":d})
(base/"CORE_REVIEWER1_F2_UNCERTAINTY.json").write_text(json.dumps({"protocol":"reviewer1_f2_uncertainty_consolidated_v1","comparisons":f2},indent=2)+"\n")
with (base/"CORE_REVIEWER1_F2_UNCERTAINTY.csv").open("w",newline="") as f:
    fields=["backbone","family","n_seeds","n_graphs","mean_o2_minus_o3","student_t_low","student_t_high","equal_graph_mean_o2_minus_o3","graph_bootstrap_low","graph_bootstrap_high","eligible_for_paper"]
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for x in f2:w.writerow({"backbone":x["backbone"],"family":x["family"],"n_seeds":x["n_seeds"],"n_graphs":x["n_graphs"],"mean_o2_minus_o3":x["mean_o2_minus_o3"],"student_t_low":x["student_t_95"][0],"student_t_high":x["student_t_95"][1],"equal_graph_mean_o2_minus_o3":x["equal_graph_mean_o2_minus_o3"],"graph_bootstrap_low":x["graph_bootstrap_95"][0],"graph_bootstrap_high":x["graph_bootstrap_95"][1],"eligible_for_paper":x["eligible_for_paper"]})
for src,dst in [("duplicate_provenance_audit.json","CORE_REVIEWER1_DUPLICATE_PROVENANCE_AUDIT.json"),("duplicate_provenance_audit.csv","CORE_REVIEWER1_DUPLICATE_PROVENANCE_AUDIT.csv"),("training_configuration.json","CORE_REVIEWER1_TRAINING_CONFIGURATION.json")]:
    (base/dst).write_bytes((raw/src).read_bytes())
print(json.dumps({"selective_per_seed_rows":len(rows),"f2_comparisons":len(f2)},indent=2))
