#!/usr/bin/env python3
"""Persistent owner for one large-model cluster large-model final-test program."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY=str(ROOT/".venv/bin/python"); TASKS=("t2","t4","t5")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser(); p.add_argument("--model-key",choices=("qwen32","llama70b"),required=True); a=p.parse_args(); model=a.model_key
subprocess.run([PY,"scripts/prepare_expansion_final_test.py","--model-key",model],cwd=ROOT,check=True)
for task in TASKS:
 mp=ROOT/f"registry/{model}_{task}_final_test_manifest.json"; mh=sha(mp); manifest=json.loads(mp.read_text())
 for shard in range(manifest["feature_shards"]):
  output=ROOT/f"outputs/{model}-{task}-final-test/cache/shard-{shard:02d}.pt"
  if not output.exists(): subprocess.run([PY,"src/precompute_expansion_test.py","--manifest",str(mp),"--manifest-sha256",mh,"--shard",str(shard)],cwd=ROOT,check=True)
 for cell in manifest["source_cells"]:
  output=ROOT/f"outputs/{model}-{task}-final-test/{cell['method']}/seed-{cell['seed']}/test_summary.json"
  if not output.exists(): subprocess.run([PY,"src/evaluate_expansion_test.py","--manifest",str(mp),"--manifest-sha256",mh,"--cell-id",cell["cell_id"]],cwd=ROOT,check=True)
subprocess.run([PY,"scripts/report_expansion_final_test.py","--models",model],cwd=ROOT,check=True)
