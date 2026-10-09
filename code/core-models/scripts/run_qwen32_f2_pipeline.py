#!/usr/bin/env python3
"""Prepare, cache, train, and report the Qwen2.5-32B 14B F2 CLadder arm."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
subprocess.run([PY,"scripts/prepare_qwen32_extension.py"],cwd=R,check=True)
m=R/"registry/qwen32_cladder_extension_manifest.json";d=json.loads(m.read_text())
if not Path(d["feature_cache_summary"]).is_file():subprocess.run([PY,"src/precompute_qwen32_cladder.py","--manifest",str(m),"--manifest-sha256",sha(m)],cwd=R,check=True)
subprocess.run([PY,"scripts/freeze_qwen32_cache.py"],cwd=R,check=True)
subprocess.run([PY,"scripts/run_qwen32_extension.py"],cwd=R,check=True)
subprocess.run([PY,"scripts/report_qwen32_extension.py"],cwd=R,check=True)
