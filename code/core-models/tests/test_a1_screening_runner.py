import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "run_a1_screening.py"
SPEC = importlib.util.spec_from_file_location("a1_screening_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class A1ScreeningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (ROOT / "registry" / "a1_screening_manifest.json").read_text()
        )

    def test_manifest_expands_to_registered_130_cells(self):
        cells = runner.validate_manifest(self.manifest, ROOT)
        self.assertEqual(len(cells), 130)
        self.assertEqual(sum(cell.family == "xor" for cell in cells), 30)
        self.assertEqual(sum(cell.arm == "t0_id" for cell in cells), 5)

    def test_tampered_seed_count_fails_registry_validation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"] = manifest["seeds"][:4]
        with self.assertRaisesRegex(ValueError, "seed count"):
            runner.validate_manifest(manifest, ROOT)

    def test_tampered_arm_mapping_fails_registry_validation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["arms"]["t0_id"]["mode"] = "t1"
        with self.assertRaisesRegex(ValueError, "implementation mapping"):
            runner.validate_manifest(manifest, ROOT)

    def test_xor_commands_include_twenty_bundles_and_frozen_eval(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "xor:t3b_pointer:2026090502"
        )
        commands = runner.commands(self.manifest, ROOT, cell)
        self.assertEqual(commands["train"].count("--xor-bundle"), 20)
        self.assertEqual(commands["composed_validation"].count("--bundle"), 20)
        self.assertIn("--hard-pointer", commands["train"])
        self.assertIn("--pointer-loss-weight", commands["train"])
        self.assertNotIn("test", commands["composed_validation"])

    def test_com2_cell_uses_open_text_training_and_adjudicated_truth(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "com2:t1_text:2026090501"
        )
        commands = runner.commands(self.manifest, ROOT, cell)
        self.assertIn("--groups-dir", commands["train"])
        self.assertIn("--paraphrase-catalog", commands["train"])
        self.assertIn("--open-text-output", commands["train"])
        self.assertIn("com2", commands["composed_validation"])
        self.assertIn("data/two_edit/manual/authoritative_manual_truth.jsonl", commands["composed_validation"])
        self.assertNotEqual(runner.cell_status(cell)["state"], "blocked_missing_authoritative_two_edit")

    def test_cladder_cell_uses_executable_truth_evaluator(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "cladder:t3b_pointer:2026090501"
        )
        commands = runner.commands(self.manifest, ROOT, cell)
        self.assertEqual(commands["composed_validation"][-3:], [
            "--bundle", "data/two_edit/cladder/artifact.json",
            "data/two_edit/cladder/manifest.jsonl",
        ])
        self.assertIn("cladder", commands["composed_validation"])
        self.assertNotEqual(
            runner.cell_status(cell)["state"],
            "blocked_missing_authoritative_two_edit",
        )

    def test_wiqa_cell_uses_signed_executable_truth_evaluator(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "wiqa:t3b_pointer:2026090501"
        )
        command = runner.commands(self.manifest, ROOT, cell)["composed_validation"]
        self.assertIn("wiqa", command)
        self.assertIn("data/two_edit/wiqa/queue.jsonl", command)
        self.assertIn("data/two_edit/wiqa/truth.jsonl", command)

    def test_ccrgb_cell_uses_executable_boolean_truth(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "ccrgb:t3b_pointer:2026090501"
        )
        command = runner.commands(self.manifest, ROOT, cell)["composed_validation"]
        self.assertIn("ccrgb", command)
        self.assertIn("data/two_edit/ccrgb/artifact.json", command)
        self.assertIn("data/two_edit/ccrgb/manifest.jsonl", command)

    def test_t0_command_is_xor_only_and_has_no_paraphrase_flag(self):
        cell = next(
            item for item in runner.validate_manifest(self.manifest, ROOT)
            if item.cell_id == "xor:t0_id:2026090501"
        )
        command = runner.commands(self.manifest, ROOT, cell)["train"]
        self.assertEqual(command[command.index("--mode") + 1], "t0")
        self.assertNotIn("--paraphrase-catalog", command)


class A1ConfirmatoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (ROOT / "registry" / "a1_confirmatory_manifest.json").read_text()
        )

    def test_manifest_expands_to_registered_520_cells(self):
        cells = runner.validate_manifest(self.manifest, ROOT)
        self.assertEqual(len(cells), 520)
        self.assertEqual(sum(cell.family == "xor" for cell in cells), 120)
        self.assertEqual(sum(cell.arm == "t0_id" for cell in cells), 20)
        self.assertTrue(all(cell.stage == "confirmatory" for cell in cells))

    def test_confirmatory_outputs_are_isolated(self):
        cells = runner.validate_manifest(self.manifest, ROOT)
        self.assertTrue(all("outputs/a1-confirmatory/" in str(cell.output) for cell in cells))

    def test_confirmatory_xor_graphs_are_disjoint_from_screening(self):
        screening = json.loads(
            (ROOT / "registry" / "a1_screening_manifest.json").read_text()
        )
        self.assertTrue(
            set(self.manifest["paths"]["xor_graphs"]).isdisjoint(
                screening["paths"]["xor_graphs"]
            )
        )
        manifest = copy.deepcopy(self.manifest)
        manifest["paths"]["xor_graphs"][0] = screening["paths"]["xor_graphs"][0]
        with self.assertRaisesRegex(ValueError, "graphs must be disjoint"):
            runner.validate_manifest(manifest, ROOT)

    def test_confirmatory_real_family_eval_uses_validation_paraphrases(self):
        cells = runner.validate_manifest(self.manifest, ROOT)
        cell = next(item for item in cells if item.cell_id == "wiqa:t3b_pointer:2026090601")
        evaluation = runner.commands(self.manifest, ROOT, cell)["composed_validation"]
        self.assertIn("--paraphrase-catalog", evaluation)
        self.assertIn("data/paraphrases/structured_catalog.json", evaluation)

    def test_confirmatory_xor_uses_intrinsic_unseen_wording(self):
        cells = runner.validate_manifest(self.manifest, ROOT)
        cell = next(item for item in cells if item.cell_id == "xor:t3b_pointer:2026090601")
        evaluation = runner.commands(self.manifest, ROOT, cell)["composed_validation"]
        self.assertNotIn("--paraphrase-catalog", evaluation)

    def test_screening_seed_reuse_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"][0] = 2026090501
        with self.assertRaisesRegex(ValueError, "disjoint"):
            runner.validate_manifest(manifest, ROOT)

    def test_screening_output_override_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["completed_cell_overrides"] = {
            "xor:t3b_pointer:2026090601": "outputs/a1-screen-t3b-xor-seed2026090501"
        }
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            runner.validate_manifest(manifest, ROOT)

    def test_frozen_selection_hash_is_enforced(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["frozen_selection"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.validate_manifest(manifest, ROOT)

    def test_registry_hash_is_enforced(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "registry hash mismatch"):
            runner.validate_manifest(manifest, ROOT)

    def test_confirmatory_boundary_forbids_test(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["data_access"]["held_out_test"] = "allowed"
        with self.assertRaisesRegex(ValueError, "data-access"):
            runner.validate_manifest(manifest, ROOT)

    def test_finalizer_distinguishes_catalog_artifact_and_semantic_hashes(self):
        source = PATH.read_text()
        self.assertIn(
            'artifact_hashes.get(catalog_ref) != _sha256(catalog_path)', source
        )
        self.assertIn(
            'configuration.get("paraphrase_catalog_sha256") != unseen.get("catalog_sha256")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
