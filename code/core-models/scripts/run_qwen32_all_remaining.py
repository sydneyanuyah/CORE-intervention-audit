#!/usr/bin/env python3
"""Single no-scheduler owner for all remaining Qwen2.5-32B experiment families."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
for script in ("scripts/run_qwen32_laws_pipeline.py","scripts/run_qwen32_remaining_pipeline.py","scripts/run_qwen32_g3_pipeline.py"):
 subprocess.run([PY,script],cwd=R,check=True)
subprocess.run([PY,"scripts/finalize_qwen32_all.py"],cwd=R,check=True)
