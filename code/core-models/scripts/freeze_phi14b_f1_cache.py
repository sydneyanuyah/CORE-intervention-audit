#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 mp=ROOT/"registry/phi14b_f1_manifest.json";m=json.loads(mp.read_text());c=Path(m["feature_cache"]);s=Path(m["feature_cache_summary"]);row=json.loads(s.read_text())
 if row["manifest_sha256"]!=sha(mp) or row["test_evaluated"] is not False: raise ValueError("cache binding mismatch")
 x={"protocol":"phi4_14b_f1_cache_binding_v1","base_manifest":str(mp),"base_manifest_sha256":sha(mp),"feature_cache":str(c),"feature_cache_sha256":sha(c),"feature_cache_summary":str(s),"feature_cache_summary_sha256":sha(s),"cells":m["cells"],"test_evaluated":False};out=ROOT/"registry/phi14b_f1_cache_amendment.json";out.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");print(json.dumps({"amendment_sha256":sha(out)}))
if __name__=="__main__":main()
