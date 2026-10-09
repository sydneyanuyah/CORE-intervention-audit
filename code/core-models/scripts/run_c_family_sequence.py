#!/usr/bin/env python3
"""Persistent automatic C1 -> C2 -> C3 -> C4 reporting handoff."""
from __future__ import annotations
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; STATUS=ROOT/"logs/c-family-sequence/status.json"
def write(stage,state,detail):
 STATUS.parent.mkdir(parents=True,exist_ok=True); STATUS.write_text(json.dumps({"stage":stage,"state":state,"detail":detail,"updated_epoch":time.time(),"test_evaluated":False},indent=2,sort_keys=True)+"\n")
def wait_c1():
 while True:
  try:d=json.loads((ROOT/"logs/c1/production/status.json").read_text())
  except (OSError,json.JSONDecodeError):d={}
  complete=int(d.get("complete",0)); active=d.get("active",{}); write("C1","waiting",f"{complete}/60 formal cells complete")
  if complete==60 and not active:return
  time.sleep(30)
def run(name):
 write(name,"running",f"generating {name} evidence"); subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/f"scripts/report_{name.lower()}.py")],cwd=ROOT,check=True); write(name,"complete",f"{name} evidence generated")
def evidence_ready(name):
 path=ROOT/f"reports/{name}_EVIDENCE.json"
 try:data=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return False
 return data.get("test_evaluated") is False and (ROOT/f"reports/{name}_REPORT.md").is_file()
def main():
 wait_c1()
 for name in ("C1","C2"):
  if evidence_ready(name): write(name,"complete",f"verified existing {name} evidence")
  else: run(name)
 if not evidence_ready("C3"):
  write("C3","running","measuring F2 XOR and repaired Com2 editor-zeroed cells")
  subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/dispatch_c3_xor.py")],cwd=ROOT,check=True)
  subprocess.run([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/dispatch_c3_com2.py")],cwd=ROOT,check=True)
 for name in ("C3","C4"):
  if evidence_ready(name): write(name,"complete",f"verified existing {name} evidence")
  else: run(name)
 write("sequence","complete","C1, C2, C3, and C4 evidence complete"); return 0
if __name__=="__main__": raise SystemExit(main())
