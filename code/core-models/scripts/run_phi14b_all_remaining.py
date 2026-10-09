#!/usr/bin/env python3
"""Single no-scheduler owner for all remaining Phi-14B experiment families."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
for script in ("scripts/run_phi14b_laws_pipeline.py","scripts/run_phi14b_remaining_pipeline.py","scripts/run_phi14b_g3_pipeline.py"):
 subprocess.run([PY,script],cwd=R,check=True)
subprocess.run([PY,"scripts/finalize_phi14b_all.py"],cwd=R,check=True)
