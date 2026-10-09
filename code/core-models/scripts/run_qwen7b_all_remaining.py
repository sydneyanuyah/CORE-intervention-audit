#!/usr/bin/env python3
"""Single no-scheduler owner for all remaining Qwen-7B experiment families."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
for script in ("scripts/run_qwen7b_laws_pipeline.py","scripts/run_qwen7b_remaining_pipeline.py","scripts/run_qwen7b_g3_pipeline.py"):
 subprocess.run([PY,script],cwd=R,check=True)
subprocess.run([PY,"scripts/finalize_qwen7b_all.py"],cwd=R,check=True)
