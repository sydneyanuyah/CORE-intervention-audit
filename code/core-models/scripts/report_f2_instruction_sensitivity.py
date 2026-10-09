#!/usr/bin/env python3
"""Aggregate registered F2 CLadder instruction-sensitivity cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = ROOT / "registry/f2_cladder_instruction_sensitivity_manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    methods = {}
    for method in manifest["methods"]:
        rows = []
        for cell in manifest["cells"]:
            if cell["method"] != method:
                continue
            path = ROOT / cell["output"] / "sensitivity_summary.json"
            row = json.loads(path.read_text())
            if row.get("manifest_sha256") != manifest_sha or row.get("test_evaluated") is not False:
                raise ValueError(f"invalid sensitivity result: {cell['cell_id']}")
            rows.append(row)
        methods[method] = {
            "cells": len(rows),
            "pair_measurements": sum(x["pair_count"] for x in rows),
            "changed_pair_vectors": sum(x["changed_pair_vectors"] for x in rows),
            "variable_decisions": sum(x["variable_decision_count"] for x in rows),
            "changed_variable_decisions": sum(x["changed_variable_decisions"] for x in rows),
            "mean_clean_two_edit_balanced": sum(x["clean_two_edit_balanced"] for x in rows) / len(rows),
            "mean_inverted_two_edit_balanced_against_retained_gold": sum(x["inverted_two_edit_balanced_against_retained_gold"] for x in rows) / len(rows),
            "mean_retained_gold_accuracy_delta": sum(x["retained_gold_accuracy_delta"] for x in rows) / len(rows),
        }
    report = {
        "protocol": manifest["protocol"], "manifest_sha256": manifest_sha,
        "completed_cells": sum(x["cells"] for x in methods.values()),
        "methods": methods, "allowed_splits": ["validation"], "test_evaluated": False,
    }
    output = ROOT / "reports/F2_INSTRUCTION_SENSITIVITY.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
