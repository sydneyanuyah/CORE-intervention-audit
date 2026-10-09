#!/usr/bin/env python3
"""Persistent no-scheduler owner for the complete Qwen2.5-32B 14B program."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];PY=str(R/".venv/bin/python")
def run(script):subprocess.run([PY,script],cwd=R,check=True)
run("scripts/prepare_qwen32_queue.py")
for script in ("scripts/prepare_qwen32_f1.py","scripts/prepare_qwen32_extension.py","scripts/prepare_qwen32_f3.py","scripts/prepare_qwen32_f4.py","scripts/prepare_qwen32_a1.py","scripts/prepare_qwen32_remaining.py"):
 run(script)
for script in ("scripts/run_qwen32_f1_pipeline.py","scripts/run_qwen32_f2_pipeline.py","scripts/run_qwen32_f3_pipeline.py","scripts/run_qwen32_f4_pipeline.py","scripts/run_qwen32_a1_pipeline.py","scripts/run_qwen32_a2_a3.py","scripts/run_qwen32_c1.py","scripts/report_qwen32_c1_c2.py","scripts/report_qwen32_base.py","scripts/run_qwen32_laws_pipeline.py","scripts/run_qwen32_remaining_pipeline.py","scripts/run_qwen32_g3_pipeline.py","scripts/finalize_qwen32_all.py"):
 run(script)
