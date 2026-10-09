#!/usr/bin/env python3
"""Extract Reviewer-1 measurements from frozen BERT result artifacts only."""

import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(xs):
    xs = np.asarray(xs, dtype=float)
    m = float(xs.mean())
    if len(xs) < 2:
        return [m, m]
    q = {5: 2.7764451051977987, 8: 2.3646242510102993,
         10: 2.2621571627409915, 15: 2.1447866879169273,
         20: 2.093024054408263}.get(len(xs))
    if q is None:
        raise RuntimeError(f"unregistered Student-t sample size: {len(xs)}")
    e = q * float(xs.std(ddof=1)) / math.sqrt(len(xs))
    return [m-e, m+e]


def selected(summary):
    rows = [x for x in summary["history"] if x.get("selected")]
    if not rows:
        raise RuntimeError("no checkpoint-selected epoch")
    # The trainer writes the checkpoint on every selected/improved row; when a
    # tied row is selected later, the last selected row is the frozen checkpoint.
    return rows[-1]["validation"]["per_variable"]["groups"]


def selective(root):
    rows=[]
    for experiment, evidence in [("L1", root/"reports/l1_inventory.json"),
                                 ("L2", root/"reports/evidence/l2_cells_complete.json")]:
        for cell in load(evidence)["cells"]:
            p=root/cell["summary"]; d=load(p)
            if sha(p) != cell["summary_sha256"] or d.get("test_evaluated") is not False:
                raise RuntimeError(f"provenance failure: {p}")
            _,family,method,seed=cell["cell_id"].split(":")
            if "history" in d:
                g=selected(d); sensitivity=g["change"]["accuracy"]; preservation=g["preservation"]["accuracy"]
            else:
                v=d["validation"]; sensitivity=v["change_accuracy"]; preservation=v["preservation_accuracy"]
            rows.append(dict(backbone="bert-base-uncased",experiment=experiment,family=family,method=method,
                seed=int(seed),cell_id=cell["cell_id"],descendant_change_sensitivity=sensitivity,
                non_descendant_preservation=preservation,selective_invariance_balanced=(sensitivity+preservation)/2,
                summary_path=str(p.relative_to(root)),summary_sha256=cell["summary_sha256"],
                checkpoint_path=cell["checkpoint"],checkpoint_sha256=cell["checkpoint_sha256"],
                split="validation",test_evaluated=False))
    ev=load(root/"reports/evidence/l3_complete.json")
    prov={x["cell_id"]:x for x in ev["provenance"]}
    for p in sorted((root/"outputs/l3").glob("*/seed-*/run_summary.json")):
        d=load(p); cell=d["cell_id"]; pr=prov[cell]
        if sha(p)!=pr["summary_sha256"] or d.get("test_evaluated") is not False:
            raise RuntimeError(f"provenance failure: {p}")
        v=d["trained"].get("single", d["trained"])
        sensitivity=v["changed_descendant_sensitivity"]; preservation=v["non_descendant_specificity"]
        rows.append(dict(backbone="bert-base-uncased",experiment="L3",family="xor",method=d["method"],seed=d["seed"],
            cell_id=cell,descendant_change_sensitivity=sensitivity,non_descendant_preservation=preservation,
            selective_invariance_balanced=(sensitivity+preservation)/2,summary_path=str(p.relative_to(root)),
            summary_sha256=pr["summary_sha256"],checkpoint_path=None,checkpoint_sha256=pr["checkpoint_sha256"],
            split="validation",test_evaluated=False))
    groups=defaultdict(list)
    for r in rows: groups[(r["backbone"],r["experiment"],r["family"],r["method"])].append(r)
    agg=[]
    for key, rr in sorted(groups.items()):
        out=dict(zip(["backbone","experiment","family","method"],key)); out["n_seeds"]=len(rr)
        for field in ["descendant_change_sensitivity","non_descendant_preservation","selective_invariance_balanced"]:
            xs=[x[field] for x in rr]; out[field+"_mean"]=float(np.mean(xs)); out[field+"_student_t_95"]=ci(xs)
        agg.append(out)
    return rows,agg


def f2(root):
    families=[]
    for family_dir in sorted((root/"outputs/f2").iterdir()):
        family=family_dir.name
        paths={m:{load(p)["seed"]:p for p in (family_dir/m).glob("seed-*/run_summary.json")} for m in ["o2","o3"] if (family_dir/m).exists()}
        if set(paths)!={"o2","o3"}: continue
        seeds=sorted(set(paths["o2"]) & set(paths["o3"]))
        # Only the formal real-family cells expose checkpoint-bound per-graph
        # measurements. Legacy/screening summaries are deliberately excluded.
        if not seeds or any("two_edit" not in load(paths[m][seeds[0]]) or
                            "per_graph" not in load(paths[m][seeds[0]])["two_edit"]
                            for m in ("o2", "o3")):
            continue
        deltas=[]; graph_deltas=defaultdict(list); provenance=[]
        for seed in seeds:
            ds={m:load(paths[m][seed]) for m in ["o2","o3"]}
            for m,d in ds.items():
                if d.get("test_evaluated") is not False: raise RuntimeError(f"test flag: {paths[m][seed]}")
                provenance.append({"seed":seed,"method":m,"cell_id":d.get("cell_id",f"f2:{family}:{m}:{seed}"),"summary_path":str(paths[m][seed].relative_to(root)),
                                   "summary_sha256":sha(paths[m][seed]),"checkpoint_sha256":d.get("checkpoint_sha256")})
            deltas.append(ds["o2"]["two_edit"]["two_edit_balanced"]-ds["o3"]["two_edit"]["two_edit_balanced"])
            g2=ds["o2"]["two_edit"]["per_graph"]; g3=ds["o3"]["two_edit"]["per_graph"]
            if set(g2)!=set(g3): raise RuntimeError(f"graph mismatch {family} {seed}")
            for g in g2: graph_deltas[g].append(g2[g]["two_edit_balanced"]-g3[g]["two_edit_balanced"])
        graph_means={g:float(np.mean(x)) for g,x in graph_deltas.items()}
        rng=np.random.default_rng(20260914); vals=np.asarray(list(graph_means.values()))
        boots=np.mean(rng.choice(vals,(10000,len(vals)),replace=True),axis=1) if len(vals) else np.asarray([])
        families.append({"backbone":"bert-base-uncased","family":family,"n_seeds":len(seeds),"n_graphs":len(vals),
                         "paired_seed_deltas_o2_minus_o3":deltas,"paired_seed_mean":float(np.mean(deltas)),
                         "paired_seed_student_t_95":ci(deltas),"equal_graph_mean":float(vals.mean()) if len(vals) else None,
                         "paired_graph_bootstrap_95":[float(x) for x in np.quantile(boots,[.025,.975])] if len(vals) else None,
                         "graph_bootstrap_replicates":10000,"split":"validation","test_evaluated":False,"provenance":provenance})
    return families


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path.cwd()); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows,agg=selective(a.root)
    with (a.output_dir/"bert_selective_invariance_per_seed.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.output_dir/"bert_selective_invariance.json").write_text(json.dumps({"protocol":"reviewer1_selective_invariance_existing_artifacts_v1","measurement":"validation only; no training; L1/L2 selected-epoch per-variable groups; L3 frozen trained-arm summaries","per_seed":rows,"aggregate":agg},indent=2)+"\n")
    (a.output_dir/"bert_f2_uncertainty.json").write_text(json.dumps({"protocol":"reviewer1_f2_uncertainty_existing_artifacts_v1","estimator":"paired seed Student-t; equal-weight graph means bootstrapped over graph IDs with seed deltas averaged within graph","families":f2(a.root)},indent=2)+"\n")
    print(json.dumps({"selective_rows":len(rows),"selective_groups":len(agg),"f2_families":len(f2(a.root))},indent=2))

if __name__=="__main__": main()
