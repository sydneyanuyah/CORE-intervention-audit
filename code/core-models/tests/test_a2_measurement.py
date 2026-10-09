import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts/run_a2_measurement.py"
SPEC = importlib.util.spec_from_file_location("run_a2_measurement", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class A2MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = runner.load(ROOT / "registry/a2_measurement_manifest.json")

    def test_manifest_expands_exact_registered_matrix(self):
        cells = runner.validate(self.manifest, ROOT)
        self.assertEqual(len(cells), 400)
        self.assertEqual(len({cell.cell_id for cell in cells}), 400)
        self.assertEqual(len({(cell.family, cell.seed) for cell in cells}), 100)

    def test_command_is_validation_only_and_forwards_control(self):
        cell = next(
            cell for cell in runner.validate(self.manifest, ROOT)
            if cell.family == "xor" and cell.control == "no_instruction_baseline"
        )
        command = runner.command(self.manifest, ROOT, cell)
        self.assertEqual(command[command.index("--split") + 1], "validation")
        self.assertEqual(command[command.index("--mode") + 1], "t2b")
        self.assertEqual(command[command.index("--a2-control") + 1], "no_instruction")
        self.assertEqual(command.count("--bundle"), 20)
        self.assertNotIn("test", command)

    def test_real_family_command_retains_paraphrase_catalog(self):
        cell = next(
            cell for cell in runner.validate(self.manifest, ROOT)
            if cell.family == "ccrgb" and cell.control == "prompting_baseline"
        )
        command = runner.command(self.manifest, ROOT, cell)
        self.assertIn("--paraphrase-catalog", command)
        self.assertEqual(command[command.index("--a2-control") + 1], "prompting")

    def test_analysis_contract_fails_closed(self):
        changed = deepcopy(self.manifest)
        changed["analysis"]["experimental_unit"] = "record"
        with self.assertRaisesRegex(ValueError, "analysis contract"):
            runner.validate(changed, ROOT)

    def test_catalog_contains_distinct_frozen_checkpoints(self):
        catalog = json.loads((ROOT / "registry/a2_source_catalog.json").read_text())
        self.assertEqual(len(catalog["cells"]), 100)
        self.assertEqual(len({row["checkpoint_sha256"] for row in catalog["cells"]}), 100)

    def test_exactly_one_active_reuse_cell_per_source(self):
        cells = runner.validate(self.manifest, ROOT)
        active = [cell for cell in cells if cell.a2_control == "active"]
        self.assertEqual(len(active), 100)
        self.assertEqual(len({(cell.family, cell.seed) for cell in active}), 100)


if __name__ == "__main__":
    unittest.main()
