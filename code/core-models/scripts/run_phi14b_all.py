#!/usr/bin/env python3
"""Persistent no-scheduler owner for the complete Phi-4 14B program."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
def run(script):subprocess.run([PY,script],cwd=R,check=True)
run("scripts/prepare_phi14b_queue.py")
for script in ("scripts/prepare_phi14b_f1.py","scripts/prepare_phi14b_extension.py","scripts/prepare_phi14b_f3.py","scripts/prepare_phi14b_f4.py","scripts/prepare_phi14b_a1.py","scripts/prepare_phi14b_remaining.py"):
 run(script)
run("scripts/run_phi14b_cache_warmup.py")
for script in ("scripts/run_phi14b_f1_pipeline.py","scripts/run_phi14b_f2_pipeline.py","scripts/run_phi14b_f3_pipeline.py","scripts/run_phi14b_f4_pipeline.py","scripts/run_phi14b_a1_pipeline.py","scripts/run_phi14b_a2_a3.py","scripts/run_phi14b_c1.py","scripts/report_phi14b_c1_c2.py","scripts/report_phi14b_base.py","scripts/run_phi14b_laws_pipeline.py","scripts/run_phi14b_remaining_pipeline.py","scripts/run_phi14b_g3_pipeline.py","scripts/finalize_phi14b_all.py"):
 run(script)
