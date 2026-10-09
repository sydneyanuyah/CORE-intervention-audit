#!/usr/bin/env python3
"""Create C3 paired editor-ablation inference from frozen measurements."""

from __future__ import annotations
import hashlib, json, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def tint(v):
    m=statistics.fmean(v); h=2.093024054*statistics.stdev(v)/math.sqrt(len(v)); return [m-h,m+h]
def boot(v):
    r=random.Random(403); x=sorted(statistics.fmean(r.choice(v) for _ in v) for _ in range(10000)); return [x[249],x[9749]]

def main():
    mp=ROOT/"registry/c3_manifest.json"; m=load(mp); zp=ROOT/m["zeroed_source"]
    if sha(zp)!=m["zeroed_source_sha256"]: raise RuntimeError("C3 zeroed source changed")
    z=load(zp)
    if z.get("completed_cells")!=400 or z.get("test_evaluated") is not False: raise RuntimeError("C3 zeroed evidence incomplete")
    catalog_path=ROOT/"registry/a2_source_catalog.json"; catalog=load(catalog_path)
    if z.get("provenance",{}).get("source_catalog_sha256")!=sha(catalog_path): raise RuntimeError("C3 zeroed/F2 source catalog binding failed")
    source_index={(row["family"],row["seed"]):row for row in catalog["cells"]}
    families={}; cells=[]
    for family in m["families"]:
        zero=(
            [float(load(ROOT / m["repaired_com2_zeroed_source"].format(seed=seed))["two_edit"]["two_edit_balanced"]) for seed in m["f2_seeds"]]
            if family == "com2" else
            [float(load(ROOT / m["xor_zeroed_source"].format(seed=seed))["two_edit"].get("two_edit_balanced",load(ROOT / m["xor_zeroed_source"].format(seed=seed))["two_edit"].get("balanced_numerical_score"))) for seed in m["f2_seeds"]]
            if family == "xor" else
            list(map(float,z["families"][family]["controls"]["t2b_editor_zeroed"]["seed_scores"]))
        )
        if len(zero)!=20: raise RuntimeError(f"C3 missing zeroed units for {family}")
        full=[]
        for i,seed in enumerate(m["f2_seeds"]):
            p=(ROOT/f"outputs/f2-fixed/com2/o3/seed-{seed}/run_summary.json") if family == "com2" else ROOT/m["full_source"].format(family=family,seed=seed); d=load(p)
            if d.get("method")!="o3" or d.get("seed")!=seed or d.get("world_size")!=4 or d.get("test_evaluated") is not False: raise RuntimeError(f"C3 invalid F2 source {p}")
            if family not in {"xor","com2"}:
                expected=source_index[(family,2026090300+seed)]["checkpoint_sha256"]
                if d.get("checkpoint_sha256")!=expected: raise RuntimeError(f"C3 F2/zeroed reader mismatch for {family}/{seed}")
            metrics=d.get("composed",d.get("two_edit",{}))
            score=float(metrics.get("two_edit_balanced",metrics.get("balanced_numerical_score",metrics.get("balanced_intervention_score"))))
            full.append(score); cells.append({"family":family,"f2_seed":seed,"a2_seed":2026090300+seed,"full":score,"edit_zeroed":zero[i],"full_summary_sha256":sha(p)})
        delta=[a-b for a,b in zip(full,zero)]
        families[family]={"full_mean":statistics.fmean(full),"edit_zeroed_mean":statistics.fmean(zero),"mean_delta":statistics.fmean(delta),"student_t_95":tint(delta),"paired_graph_bootstrap_95":boot(delta),"unit_count":20}
    out={"protocol":m["protocol"],"manifest_sha256":sha(mp),"families":families,"cells":cells,"cell_count":100,"test_evaluated":False}
    (ROOT/"reports/C3_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    lines=["# C3 F2 editor ablation","","Validation-only paired inference; no held-out test access.","","| Family | Full O3 | Editor zeroed | Delta | 95% t interval | 95% bootstrap |","|---|---:|---:|---:|---:|---:|"]
    for f,d in families.items(): lines.append(f"| {f} | {d['full_mean']:.6f} | {d['edit_zeroed_mean']:.6f} | {d['mean_delta']:+.6f} | [{d['student_t_95'][0]:+.6f}, {d['student_t_95'][1]:+.6f}] | [{d['paired_graph_bootstrap_95'][0]:+.6f}, {d['paired_graph_bootstrap_95'][1]:+.6f}] |")
    (ROOT/"reports/C3_REPORT.md").write_text("\n".join(lines)+"\n"); return 0

if __name__=="__main__": raise SystemExit(main())
