"""Inspect pinned CSuite archives without opening released test payloads.

The released interventional arrays are explicitly described as test data.  This
inspector therefore limits itself to archive identities, graph metadata, and
the released train/validation file inventory.  A later CORE adapter must
generate fresh development interventions from the pinned simulator rather than
selecting against the released interventional test arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


PINNED_REVISION = "not-published"
VERSION = "v0.1"
ARCHIVE_HASHES = {
    "cat_chain": "not-published",
    "cat_collider": "not-published",
    "cat_to_cts": "not-published",
    "cts_to_cat": "not-published",
    "large_backdoor_binary_t": "not-published",
    "large_backdoor": "not-published",
    "linexp": "not-published",
    "lingauss": "not-published",
    "mixed_confounding": "not-published",
    "mixed_simpson": "not-published",
    "nonlin_simpson": "not-published",
    "nonlingauss": "not-published",
    "symprod_simpson": "not-published",
    "weak_arrows_binary_t": "not-published",
    "weak_arrows": "not-published",
}
FORBIDDEN_MEMBERS = frozenset({"test.csv", "interventions.json", "counterfactuals.json"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_development_member(archive: zipfile.ZipFile, name: str) -> bytes:
    if name in FORBIDDEN_MEMBERS or name.startswith("test"):
        raise ValueError(f"released test member is sealed: {name}")
    return archive.read(name)


def inspect(root: Path) -> dict:
    datasets = []
    for name, expected_hash in sorted(ARCHIVE_HASHES.items()):
        path = root / f"csuite_{name}_{VERSION}.zip"
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{name}: archive SHA-256 mismatch")
        with zipfile.ZipFile(path) as archive:
            members = {item.filename: item.file_size for item in archive.infolist()}
            if not {"adj_matrix.csv", "train.csv", "val.csv", "variables.json"} <= set(members):
                raise ValueError(f"{name}: required development metadata is missing")
            adjacency = list(csv.reader(io.StringIO(
                read_development_member(archive, "adj_matrix.csv").decode("utf-8")
            )))
            variables = json.loads(read_development_member(archive, "variables.json"))
        node_count = len(adjacency)
        if node_count == 0 or any(len(row) != node_count for row in adjacency):
            raise ValueError(f"{name}: adjacency matrix is not square")
        edge_count = sum(int(value) for row in adjacency for value in row)
        datasets.append({
            "dataset": name,
            "archive": str(path),
            "archive_sha256": actual_hash,
            "node_count": node_count,
            "edge_count": edge_count,
            "variable_count": len(variables.get("variables", [])),
            "development_members": {
                key: members[key] for key in ("train.csv", "val.csv", "adj_matrix.csv", "variables.json")
            },
            "sealed_members_present": sorted(FORBIDDEN_MEMBERS & set(members)),
        })
    return {
        "source": "csuite",
        "release": VERSION,
        "repository_revision": PINNED_REVISION,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "test_evaluated": False,
        "development_policy": (
            "use released train/val metadata only; generate fresh development interventions "
            "from the pinned official simulator; never read released test/intervention arrays"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/csuite"))
    parser.add_argument("--summary", type=Path, default=Path("reports/csuite_schema_summary.json"))
    args = parser.parse_args()
    summary = inspect(args.root)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset_count": summary["dataset_count"],
        "test_evaluated": summary["test_evaluated"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
