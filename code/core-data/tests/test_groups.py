import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groups import group_entry, stable_group_digest, write_group_manifest  # noqa: E402
from schema import load_jsonl  # noqa: E402
from verify import verify  # noqa: E402


class GroupSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_jsonl(ROOT / "tests" / "fixtures" / "valid_records.jsonl")

    def make_priority_tree(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        records = root / "records"
        splits = root / "splits"
        groups = root / "groups"
        records.mkdir()
        splits.mkdir()
        rows = []
        for index, fixture in enumerate(self.fixtures):
            row = copy.deepcopy(fixture)
            row["id"] = f"cladder-{index}"
            row["source"] = "cladder"
            rows.append(row)
        (records / "cladder.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (splits / "cladder.train.txt").write_text(rows[0]["id"] + "\n", encoding="utf-8")
        (splits / "cladder.validation.txt").write_text(rows[1]["id"] + "\n", encoding="utf-8")
        (splits / "cladder.test.txt").write_text("", encoding="utf-8")
        return temporary, records, splits, groups, rows

    def test_manifest_write_is_sorted_and_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            entries = [
                group_entry("b", "graph-b", "world-b"),
                group_entry("a", "graph-a", "world-a"),
            ]
            write_group_manifest(path, entries)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["record_id"] for row in rows], ["a", "b"])
            self.assertEqual(
                stable_group_digest("source", ["root", "child"]),
                stable_group_digest("source", ["root", "child"]),
            )

    def test_verifier_requires_one_mapping_per_priority_record(self):
        temporary, records, splits, groups, rows = self.make_priority_tree()
        self.addCleanup(temporary.cleanup)
        write_group_manifest(
            groups / "cladder.jsonl",
            [group_entry(rows[0]["id"], "graph-0", "world-0")],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = verify(records, splits, groups_dir=groups)
        self.assertEqual(status, 1)
        self.assertIn("lack a group mapping", output.getvalue())

    def test_verifier_rejects_group_crossing_splits(self):
        temporary, records, splits, groups, rows = self.make_priority_tree()
        self.addCleanup(temporary.cleanup)
        write_group_manifest(
            groups / "cladder.jsonl",
            [
                group_entry(rows[0]["id"], "shared-graph", "world-0"),
                group_entry(rows[1]["id"], "shared-graph", "world-1"),
            ],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = verify(records, splits, groups_dir=groups)
        self.assertEqual(status, 1)
        self.assertIn("graph group ids cross splits", output.getvalue())

    def test_verifier_rejects_world_group_crossing_splits(self):
        temporary, records, splits, groups, rows = self.make_priority_tree()
        self.addCleanup(temporary.cleanup)
        write_group_manifest(
            groups / "cladder.jsonl",
            [
                group_entry(rows[0]["id"], "graph-0", "shared-world"),
                group_entry(rows[1]["id"], "graph-1", "shared-world"),
            ],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = verify(records, splits, groups_dir=groups)
        self.assertEqual(status, 1)
        self.assertIn("world group ids cross splits", output.getvalue())

    def test_verifier_rejects_multiple_mappings_for_one_record(self):
        temporary, records, splits, groups, rows = self.make_priority_tree()
        self.addCleanup(temporary.cleanup)
        write_group_manifest(
            groups / "cladder-a.jsonl",
            [
                group_entry(rows[0]["id"], "graph-0", "world-0"),
                group_entry(rows[1]["id"], "graph-1", "world-1"),
            ],
        )
        write_group_manifest(
            groups / "cladder-b.jsonl",
            [group_entry(rows[0]["id"], "graph-0", "world-0")],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = verify(records, splits, groups_dir=groups)
        self.assertEqual(status, 1)
        self.assertIn("multiple group mappings", output.getvalue())

    def test_verifier_accepts_complete_isolated_groups(self):
        temporary, records, splits, groups, rows = self.make_priority_tree()
        self.addCleanup(temporary.cleanup)
        write_group_manifest(
            groups / "cladder.jsonl",
            [
                group_entry(rows[0]["id"], "graph-0", "world-0"),
                group_entry(rows[1]["id"], "graph-1", "world-1"),
            ],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = verify(records, splits, groups_dir=groups)
        self.assertEqual(status, 0, output.getvalue())
        self.assertIn("priority_group_coverage=complete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
