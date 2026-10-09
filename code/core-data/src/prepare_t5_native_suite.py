"""Prepare T5 as a suite of source-native causal tasks without schema coercion."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

from convert_com2 import SOURCE_SHA256 as COM2_SHA, sha256_file
from convert_counterbench import FILES as COUNTERBENCH_FILES
from convert_crass import SHA256 as CRASS_SHA
from prepare_corr2cause_t3 import RAW_HASHES as CORR_HASHES


PROTOCOL = "t5_native_transfer_suite_v1"


def stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(f"t5-native-v1:{group}".encode()).hexdigest()[:8], 16) % 100
    # A 75/10/15 native split keeps the generated held-out partition substantial
    # (well above 400 rows) while leaving the official Corr2Cause dev set untouched.
    return "train" if bucket < 75 else "validation" if bucket < 85 else "test"


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True); digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line); digest.update(line.encode())
    return digest.hexdigest()


def crass_rows(path: Path) -> list[dict]:
    if sha256_file(path) != CRASS_SHA: raise ValueError("CRASS hash mismatch")
    with tarfile.open(path, "r:gz") as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("/CRASS_FTM_main_data_set.csv"))
        rows = list(csv.DictReader(io.StringIO(archive.extractfile(member).read().decode("utf-8-sig")), delimiter=";"))
    output = []
    for row in rows:
        group = f"crass:{row['BatchID']}:{row['PCTID']}"
        output.append({"id": group, "protocol": PROTOCOL, "source": "crass", "native_task": "counterfactual_multiple_choice", "split": stable_split(group), "group_id": group, "premise": row["Premise"], "question": row["QCC"], "options": [value for value in (row["Answer1"], row["Answer2"], row["PossibleAnswer3"]) if value], "answer": row["CorrectAnswer"], "test_evaluated": False})
    return output


def counterbench_rows(root: Path) -> list[dict]:
    output = []
    for filename in ("data_balanced_alpha_V1.json", "data_balanced_backdoor_V2.json"):
        path = root / filename
        if sha256_file(path) != COUNTERBENCH_FILES[filename]: raise ValueError(f"{filename} hash mismatch")
        for index, row in enumerate(json.loads(path.read_text())):
            meta = row["meta"]
            group = f"counterbench:{filename}:{meta['graph_id']}:{meta['model_id']}:{meta['story_id']}"
            output.append({"id": f"{group}:{row['question_id']}:{index}", "protocol": PROTOCOL, "source": "counterbench", "native_task": "scm_counterfactual_binary", "split": stable_split(group), "group_id": group, "context": row["given_info"], "question": row["question"], "answer": row["answer"], "rung": meta["rung"], "query_type": meta["query_type"], "test_evaluated": False})
    return output


def corr2cause_rows(path: Path) -> list[dict]:
    if sha256_file(path) != CORR_HASHES["clean"]: raise ValueError("Corr2Cause hash mismatch")
    with path.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    return [{"id": f"corr2cause-dev:{index:04d}", "protocol": PROTOCOL, "source": "corr2cause", "native_task": "causal_relation_binary", "split": "validation", "group_id": f"corr2cause:{row['num_variables']}:{row['template']}", "input": row["input"], "answer": int(row["label"]), "test_evaluated": False} for index, row in enumerate(rows)]


def com2_rows(path: Path) -> list[dict]:
    if sha256_file(path) != COM2_SHA: raise ValueError("Com2 hash mismatch")
    output = []
    for index, row in enumerate(json.loads(path.read_text())):
        group = f"com2:{hashlib.sha256(row['chains'][0][0].encode()).hexdigest()[:20]}" if row.get("chains") else f"com2:{index:04d}"
        output.append({"id": f"com2:{index:04d}", "protocol": PROTOCOL, "source": "com2", "native_task": row["type"], "split": stable_split(group), "group_id": group, "scenario": row.get("scenario"), "question": row.get("question"), "options": row.get("options"), "answer": row.get("answer"), "chains": row.get("chains"), "test_evaluated": False})
    return output


def prepare(raw_root: Path, output: Path, summary_path: Path) -> dict:
    rows = []
    rows.extend(crass_rows(raw_root / "crass/crass-3944517b.tar.gz"))
    rows.extend(counterbench_rows(raw_root / "counterbench"))
    rows.extend(corr2cause_rows(raw_root / "corr2cause/dev.csv"))
    rows.extend(com2_rows(raw_root / "com2/main.json"))
    digest = write_jsonl(output, rows)
    summary = {
        "protocol": PROTOCOL, "rows": len(rows),
        "rows_by_source": dict(sorted(collections.Counter(row["source"] for row in rows).items())),
        "rows_by_split": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "groups_by_source": {source: len({row["group_id"] for row in rows if row["source"] == source}) for source in sorted({row["source"] for row in rows})},
        "output_sha256": digest,
        "scope": "source-native tasks; no coercion into the CORE paired-state schema",
        "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/t5/native_suite.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/t5_native_suite_summary.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_root, args.output, args.summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
