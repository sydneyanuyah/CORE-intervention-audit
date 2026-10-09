#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MP=ROOT/"registry/t2_manifest.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads(MP.read_text()); rows=[]
 for seed in m["seeds"]:
  p=ROOT/f"outputs/t2/seed-{seed}/run_summary.json"; d=json.loads(p.read_text())
  if d.get("test_evaluated") is not False or d.get("manifest_sha256")!=sha(MP) or d.get("world_size")!=4: raise ValueError(f"T2 provenance failed: {seed}")
  rows.append({"seed":seed,"summary_sha256":sha(p),"validation":d["validation"],"by_sem_family":d["by_sem_family"]})
 vals=[r["validation"]["balanced_direction_accuracy"] for r in rows]; mean=statistics.fmean(vals); half=2.776445105*statistics.stdev(vals)/math.sqrt(len(vals))
 fams={f:{"mean_balanced_direction_accuracy":statistics.fmean(r["by_sem_family"][f]["balanced_direction_accuracy"] for r in rows)} for f in rows[0]["by_sem_family"]}
 out={"protocol":m["protocol"],"manifest_sha256":sha(MP),"seeds":rows,"aggregate":{"mean_balanced_direction_accuracy":mean,"student_t_95":[mean-half,mean+half]},"by_sem_family":fams,"test_evaluated":False}
 (ROOT/"reports/T2_EVIDENCE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 lines=["# T2 CSuite held-out-family transfer","",f"Five four-GPU BERT-base/O3 cells completed. Mean held-out balanced direction accuracy: **{mean:.6f}** (95% CI [{mean-half:.6f}, {mean+half:.6f}]).","","| Held-out SEM family | Balanced direction accuracy |","|---|---:|"]
 for f,d in fams.items(): lines.append(f"| {f} | {d['mean_balanced_direction_accuracy']:.6f} |")
 (ROOT/"reports/T2_REPORT.md").write_text("\n".join(lines)+"\n"); return 0
if __name__=="__main__": raise SystemExit(main())
