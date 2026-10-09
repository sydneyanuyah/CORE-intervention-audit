#!/usr/bin/env python3
"""Freeze post-A1 confirmatory, A2, and A3 manifests from exact upstream bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def dump_once(path: Path, value) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def confirmatory() -> Path:
    template = load(ROOT / "registry/a1_repaired_confirmatory_template.json")
    report_path = ROOT / "reports/A1_REPAIRED_SCREENING.json"
    screening_path = ROOT / "registry/a1_repaired_screening_manifest.json"
    report = load(report_path)
    template["registry_sha256"] = sha(ROOT / template["registry"])
    template["frozen_selection"].update({
        "sha256": sha(report_path),
        "selected_arm": report["selected_arm"],
        "manifest_sha256": sha(screening_path),
    })
    output = ROOT / "registry/a1_repaired_confirmatory_manifest.json"
    dump_once(output, template)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("confirmatory",))
    args = parser.parse_args()
    path = confirmatory()
    print(json.dumps({"manifest": str(path), "sha256": sha(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
