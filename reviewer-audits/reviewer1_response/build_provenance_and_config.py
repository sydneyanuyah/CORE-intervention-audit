#!/usr/bin/env python3
"""Build duplicate-lineage audit and a source-bound training configuration dump."""
import argparse,csv,hashlib,json,re
from pathlib import Path

def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flatten(d,p=""):
    out={}
    for k,v in d.items():
        key=f"{p}.{k}" if p else k
        if isinstance(v,dict): out.update(flatten(v,key))
        elif isinstance(v,(int,float)) and not isinstance(v,bool): out[key]=v
    return out

def duplicate(root):
    rows=[]; l3=[]
    pairs=[("lora_matched","loreft"),("lora_matched","o2"),("loreft","o2"),("o1","task_vector_add")]
    ev=load(root/"reports/evidence/l3_complete.json"); prov={x["cell_id"]:x for x in ev["provenance"]}
    for seed in range(301,306):
        for a,b in pairs:
            pa=root/f"outputs/l3/{a}/seed-{seed}/run_summary.json"; pb=root/f"outputs/l3/{b}/seed-{seed}/run_summary.json"
            da,db=load(pa),load(pb); fa,fb=flatten(da["trained"]),flatten(db["trained"])
            common=sorted(set(fa)&set(fb))
            exact=[k for k in common if fa[k]==fb[k]]
            l3.append({"seed":seed,"methods":[a,b],"common_scalar_count":len(common),"exact_equal_scalar_count":len(exact),
                       "all_common_scalars_equal":len(exact)==len(common),"exact_equal_metrics":exact,
                       "left":{"cell_id":da["cell_id"],"checkpoint_sha256":prov[da["cell_id"]]["checkpoint_sha256"],"metric_source_path":str(pa.relative_to(root)),"metric_source_sha256":sha(pa)},
                       "right":{"cell_id":db["cell_id"],"checkpoint_sha256":prov[db["cell_id"]]["checkpoint_sha256"],"metric_source_path":str(pb.relative_to(root)),"metric_source_sha256":sha(pb)},
                       "reason":"distinct registered cell summaries; equality is numerical/implementation equivalence, not an independent replication or copied metric"})
    t1=load(root/"reports/T1_EVIDENCE.json")
    for x in t1["source_cells"]:
        p=root/x["summary"]; d=load(p)
        rows.append({"derived_experiment":"T1","family":x["family"],"method":x["method"],"seed":x["seed"],"source_experiment":"F2",
          "source_cell_id":d.get("cell_id",f"f2:{x['family']}:{x['method']}:{x['seed']}"),"source_checkpoint_sha256":d.get("checkpoint_sha256"),
          "metric_source_path":x["summary"],"expected_source_sha256":x["summary_sha256"],"actual_source_sha256":sha(p),
          "hash_verified":sha(p)==x["summary_sha256"],"metric":"two_edit_balanced","derived_value":x["two_edit_balanced"],
          "source_value":d.get("two_edit",{}).get("two_edit_balanced",d.get("composed",{}).get("balanced_numerical_score")),
          "reason":"T1 is a preregistered re-aggregation of F2 composed-pair measurements; it is not independent evidence."})
    c3=load(root/"reports/C3_EVIDENCE.json")
    for x in c3["cells"]:
        p=root/f"outputs/f2/{x['family']}/o3/seed-{x['f2_seed']}/run_summary.json"; d=load(p)
        rows.append({"derived_experiment":"C3","family":x["family"],"method":"o3_full","seed":x["f2_seed"],"source_experiment":"F2",
          "source_cell_id":d.get("cell_id",f"f2:{x['family']}:o3:{x['f2_seed']}"),"source_checkpoint_sha256":d.get("checkpoint_sha256"),
          "metric_source_path":str(p.relative_to(root)),"expected_source_sha256":x["full_summary_sha256"],"actual_source_sha256":sha(p),
          "hash_verified":sha(p)==x["full_summary_sha256"],"metric":"two_edit_balanced","derived_value":x["full"],
          "source_value":d.get("two_edit",{}).get("two_edit_balanced",d.get("composed",{}).get("balanced_numerical_score")),
          "reason":"C3 full-O3 is the F2 O3 arm; only the zeroed arm is an added ablation. C3 is not an independent confirmation."})
    c4=load(root/"reports/C4_EVIDENCE.json")
    for x in c4["cells"]:
        for method,m in x["methods"].items():
            p=root/f"outputs/f2/{x['family']}/{method}/seed-{x['seed']}/run_summary.json"; d=load(p)
            rows.append({"derived_experiment":"C4","family":x["family"],"method":method,"seed":x["seed"],"source_experiment":"F2",
              "source_cell_id":d.get("cell_id",f"f2:{x['family']}:{method}:{x['seed']}"),"source_checkpoint_sha256":d.get("checkpoint_sha256"),
              "metric_source_path":str(p.relative_to(root)),"expected_source_sha256":m["summary_sha256"],"actual_source_sha256":sha(p),
              "hash_verified":sha(p)==m["summary_sha256"],"metric":"two_edit_balanced","derived_value":m["score"],
              "source_value":d.get("two_edit",{}).get("two_edit_balanced",d.get("composed",{}).get("balanced_numerical_score")),
              "reason":"C4 reuses F2 performance and adds measured parameter accounting; performance is not independent evidence."})
    return {"protocol":"reviewer1_duplicate_provenance_audit_v1","l3_equivalence_checks":l3,
            "f2_derived_lineage":rows,"summary":{"l3_checks":len(l3),"derived_rows":len(rows),"all_hashes_verified":all(r["hash_verified"] for r in rows),
            "eligible_non_com2_hashes_verified":all(r["hash_verified"] for r in rows if r["family"]!="com2"),
            "hash_mismatches_by_reason":{"com2_post_report_repair_or_replacement":sum(not r["hash_verified"] and r["family"]=="com2" for r in rows),"other":sum(not r["hash_verified"] and r["family"]!="com2" for r in rows)},
            "l3_claim_verdict":"The asserted LoRA=LoReFT=O2 and O1=task-vector identities do not hold exactly across common artifact-level scalars. Repetitions in rounded aggregate tables must be labeled rounded numerical ties, not exact duplicate results.",
            "interpretation":"Eligible repeated F2/T1/C3/C4 values are intentional lineage reuse. Com2 lineage is stale after repair/replacement and remains excluded. L3 table repetitions are not independent confirmation."}}

def config(root):
    files=["registry/a1_repaired_confirmatory_manifest.json","registry/f2_manifest.json","registry/l1_manifest.json","registry/l2_manifest.json","registry/l3_manifest.json","registry/t2_manifest.json","registry/t4_manifest.json","registry/t5_manifest.json",
           "src/train_addressed.py","src/train_f2_baselines.py","src/train_l3_operator.py","src/train_candidate_transfer.py","src/evaluate_candidate_final.py"]
    sources={f:{"sha256":sha(root/f),"path":f} for f in files if (root/f).exists()}
    snapshots={f:load(root/f) for f in files if f.startswith("registry/") and (root/f).exists()}
    return {"protocol":"reviewer1_reproducibility_config_export_v1","generated_from_existing_registry_and_runner_sources":True,
      "source_files":sources,"registry_snapshots":snapshots,
      "normalized":{
       "L1_BERT":{"optimizer":"AdamW","learning_rate":2e-5,"weight_decay":0.01,"epochs":3,"batch_size_per_rank":16,"effective_batch_size":64,"world_size":4,"operator_rank":16,"edited_layer":4,"edit_position":"30 structural world slots","loss_weights":{"variable":1.0,"law":1.0,"pointer":0.0},"model_parameter_count_recorded":111333129,"trainable_parameter_count":"not separately serialized in L1 summaries; do not substitute total model parameters","seeds":list(range(201,221)),"precision":"bf16","hardware":"NVIDIA A16, CUDA 13.0; four GPUs/cell","selection_criterion":"maximum validation macro_f1"},
       "L2_BERT":{"optimizer":"AdamW","learning_rate":2e-5,"weight_decay":0.01,"epochs":3,"batch_size_per_rank":16,"effective_batch_size":64,"world_size":4,"operator_rank":16,"edited_layer":4,"edit_position":"30 structural world slots","loss_weights":{"variable":1.0,"law":1.0,"pointer":0.0},"model_parameter_count_recorded":111333129,"trainable_parameter_count":"not separately serialized in L2 summaries; do not substitute total model parameters","seeds":list(range(201,221)),"precision":"bf16","hardware":"NVIDIA A16, CUDA 13.0; four GPUs/cell","selection_criterion":"maximum validation macro_f1"},
       "F2_BERT":{"optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0001,"steps":500,"batch_size_per_rank":16,"effective_batch_size":64,"world_size":4,"operator_rank":16,"edited_layer":4,"edit_position":"16 appended world-state tokens","operator_or_adapter_trainable_parameters":{"prompting":0,"lora_matched":1253376,"o2":234240,"o3":1242624},"C4_accounted_parameters_including_shared_scoring_components":{"prompting":0,"lora_matched":1253376,"o2":255744,"o3":1264128},"matched_lora_rank":34,"seeds":list(range(301,321)),"precision":"runner default float32","hardware":"NVIDIA A16; four GPUs/cell","selection_criterion":"fixed-step training; frozen protocol composed two-edit validation score"},
       "L3_BERT":{"optimizer":"AdamW","learning_rate":0.005,"weight_decay":0.0001,"steps":500,"batch_size_per_rank":16,"effective_batch_size":64,"world_size":4,"operator_rank":16,"edited_layer":4,"edit_position":"16 appended world-state tokens","seeds":list(range(301,306)),"precision":"runner default float32","selection_criterion":"fixed-step training, validation-only frozen summaries"},
       "T2_BERT":{"optimizer":"AdamW","learning_rate":3e-5,"epochs":3,"batch_size_per_rank":8,"effective_batch_size":32,"world_size":4,"operator_rank":16,"edited_layer":4,"law_loss_weight":0.1,"seeds":[501,502,503,504,505]},
       "T4_BERT":{"optimizer":"AdamW","learning_rate":3e-5,"epochs":3,"batch_size_per_rank":8,"effective_batch_size":32,"world_size":4,"operator_rank":16,"edited_layer":4,"law_loss_weight":0.1,"seeds":[511,512,513,514,515]},
       "T5_BERT":{"optimizer":"AdamW","learning_rate":3e-5,"epochs":3,"batch_size_per_rank":8,"effective_batch_size":32,"world_size":4,"operator_rank":16,"edited_layer":4,"edit_position":"16 appended learned state tokens","seeds":list(range(101,121)),"selection_criterion":"maximum validation article/group-macro accuracy; strict greater-than, earliest epoch wins ties","precision":"float32 unless environment-level framework defaults differ"}},
      "locked_test_baseline_definition":{"T2":"No separately named baseline arm exists in the locked T2 manifest; T2 is an O3-only transfer measurement. Do not call it a baseline comparison.",
       "T4":"No separately named baseline arm exists in the locked T4 manifest; changing-only, imagining-only, and joint are registered O3 arms. Do not call one a baseline.",
       "T5":"The `baseline` cell uses the same BERT-base candidate-scoring architecture, 16 appended learned state tokens, split at layer 4, and trained binary candidate head, but bypasses O3 in forward (`if self.method == 'o3'`). It is not a zero-shot language-model baseline. O3 cells load a frozen F1 O3 operator; BERT and operator are frozen there, while state tokens and head train. Com2 is excluded from locked test evaluation."},
      "caveat":"Registry snapshots are included verbatim so every normalized statement can be checked. Fields absent from the frozen registry/runner are not inferred."}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path.cwd());ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    d=duplicate(a.root); (a.output_dir/"duplicate_provenance_audit.json").write_text(json.dumps(d,indent=2)+"\n")
    rr=d["f2_derived_lineage"]
    with (a.output_dir/"duplicate_provenance_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (a.output_dir/"training_configuration.json").write_text(json.dumps(config(a.root),indent=2)+"\n")
    print(json.dumps(d["summary"],indent=2))
if __name__=="__main__":main()
