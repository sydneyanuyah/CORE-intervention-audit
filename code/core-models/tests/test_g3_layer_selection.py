import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "run_g3_layer_selection.py"
SPEC = importlib.util.spec_from_file_location("g3_layer_selection", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class G3LayerSelectionTests(unittest.TestCase):
    def manifest(self):
        return json.loads((ROOT / "registry" / "g3_layer_selection_manifest.json").read_text())

    def test_manifest_expands_exhaustive_valid_boundaries(self):
        cells = runner.validate_manifest(self.manifest(), ROOT)
        self.assertEqual([cell.layer for cell in cells], list(range(1, 12)))
        self.assertEqual(len({cell.cell_id for cell in cells}), 11)

    def test_commands_are_four_gpu_launcher_validation_only(self):
        manifest = self.manifest()
        cell = runner.validate_manifest(manifest, ROOT)[0]
        commands = runner.commands(manifest, ROOT, cell)
        self.assertEqual(commands["train"][commands["train"].index("--split-layer") + 1], "1")
        self.assertEqual(commands["validation"][commands["validation"].index("--split") + 1], "validation")
        self.assertNotIn("test", " ".join(commands["train"] + commands["validation"]))

    def test_confirmatory_graph_reuse_fails(self):
        manifest = self.manifest()
        confirmatory = json.loads((ROOT / "registry" / "a1_confirmatory_manifest.json").read_text())
        manifest["paths"]["xor_graphs"] = confirmatory["paths"]["xor_graphs"]
        with self.assertRaisesRegex(ValueError, "screening development graphs"):
            runner.validate_manifest(manifest, ROOT)

    def test_tampered_registry_hash_fails(self):
        manifest = copy.deepcopy(self.manifest())
        manifest["registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "registry hash"):
            runner.validate_manifest(manifest, ROOT)

    def test_report_tie_breaks_to_smallest_layer(self):
        rows = [
            {"test_evaluated": False, "layer": layer, "two_edit_balanced": 0.5}
            for layer in range(1, 12)
        ]
        original = runner._load
        runner._load = lambda path: rows[int(path.parts[-3].split("-")[-1]) - 1]
        try:
            value = runner.report(runner.expand_cells(self.manifest(), ROOT))
        finally:
            runner._load = original
        self.assertEqual(value["selected_layer"], 1)


if __name__ == "__main__":
    unittest.main()
