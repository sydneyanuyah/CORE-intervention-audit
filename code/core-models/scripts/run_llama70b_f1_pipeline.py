#!/usr/bin/env python3
"""F1 cache, immutable binding, then GPU dispatch."""
import hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 py=str(R/".venv/bin/python");subprocess.run([py,"scripts/prepare_llama70b_f1.py"],cwd=R,check=True);m=R/"registry/llama70b_f1_manifest.json"
 if not (R/"outputs/llama70b-f1/cache/cache_summary.json").exists():subprocess.run([py,"src/precompute_llama70b_f1.py","--manifest",str(m),"--manifest-sha256",sha(m)],cwd=R,check=True)
 subprocess.run([py,"scripts/freeze_llama70b_f1_cache.py"],cwd=R,check=True);subprocess.run([py,"scripts/run_llama70b_f1.py"],cwd=R,check=True)
if __name__=="__main__":main()
