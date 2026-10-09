import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts" / "launch_addressed_training.py"
SPEC = importlib.util.spec_from_file_location("addressed_training_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class AddressedTrainingLauncherTests(unittest.TestCase):
    def args(self, size="base", *extra):
        return launcher.parse_args([
            "--mode", "t2b", "--model-size", size,
            "--data-root", "/data", "--output-dir", "/outputs", *extra,
        ])

    def test_gpu_policies(self):
        self.assertEqual(launcher.MODELS["base"]["gpus"], 4)
        self.assertEqual(launcher.MODELS["large"]["gpus"], 8)
        self.assertIn("--nproc_per_node=4", launcher.build_command(self.args("base")))
        self.assertIn("--nproc_per_node=8", launcher.build_command(self.args("large")))

    def test_seed_is_forwarded(self):
        command = launcher.build_command(self.args("base", "--seed", "2026090505"))
        index = command.index("--seed")
        self.assertEqual(command[index + 1], "2026090505")

    def test_split_layer_is_forwarded_and_non_negative(self):
        command = launcher.build_command(self.args("base", "--split-layer", "3"))
        index = command.index("--split-layer")
        self.assertEqual(command[index + 1], "3")
        with self.assertRaises(SystemExit):
            self.args("base", "--split-layer", "-1")

    def test_all_current_arms_and_resume(self):
        self.assertEqual(launcher.MODES, ("t0", "t1", "t2a", "t2b", "t3a", "t3b"))
        command = launcher.build_command(self.args("base", "--resume", "/checkpoints/last.pt"))
        self.assertEqual(command[-2:], ["--resume", "/checkpoints/last.pt"])

    def test_t0_is_xor_only(self):
        with self.assertRaises(SystemExit):
            launcher.parse_args([
                "--mode", "t0", "--model-size", "base",
                "--data-root", "/data", "--output-dir", "/output",
                "--sources", "wiqa",
            ])
        args = launcher.parse_args([
            "--mode", "t0", "--model-size", "base",
            "--data-root", "/data", "--output-dir", "/output",
            "--sources", "xor", "--xor-bundle", "/graph", "/pairs.jsonl",
        ])
        self.assertIn("t0", launcher.build_command(args))

    def test_frozen_checkpoint_measurement_is_forwarded(self):
        args = launcher.parse_args([
            "--mode", "t3b", "--model-size", "base",
            "--data-root", "/data", "--output-dir", "/output",
            "--measure-checkpoint", "/frozen/best.pt",
            "--span-override", "random",
        ])
        command = launcher.build_command(args)
        self.assertIn("--measure-checkpoint", command)
        self.assertIn("/frozen/best.pt", command)
        self.assertIn("random", command)

    def test_prediction_export_requires_measurement_and_is_forwarded(self):
        with self.assertRaises(SystemExit):
            self.args("base", "--prediction-jsonl", "/outputs/predictions.jsonl")
        args = self.args(
            "base", "--measure-checkpoint", "/frozen/best.pt",
            "--prediction-jsonl", "/outputs/predictions.jsonl",
        )
        command = launcher.build_command(args)
        index = command.index("--prediction-jsonl")
        self.assertEqual(command[index + 1], "/outputs/predictions.jsonl")

    def test_resume_and_measurement_are_exclusive(self):
        with self.assertRaises(SystemExit):
            self.args(
                "base", "--resume", "/run/last.pt",
                "--measure-checkpoint", "/run/best.pt",
            )

    def test_optional_variable_loss_is_forwarded_and_non_negative(self):
        args = self.args("base", "--variable-loss-weight", "0.25")
        command = launcher.build_command(args)
        index = command.index("--variable-loss-weight")
        self.assertEqual(command[index + 1], "0.25")
        with self.assertRaises(SystemExit):
            self.args("base", "--variable-loss-weight", "-0.1")
        with self.assertRaises(SystemExit):
            self.args("base", "--variable-loss-weight", "nan")

    def test_pointer_supervision_is_t3b_only_and_forwarded(self):
        args = launcher.parse_args([
            "--mode", "t3b", "--model-size", "base",
            "--data-root", "/data", "--output-dir", "/outputs",
            "--pointer-loss-weight", "1.0",
        ])
        command = launcher.build_command(args)
        index = command.index("--pointer-loss-weight")
        self.assertEqual(command[index + 1], "1.0")
        with self.assertRaises(SystemExit):
            self.args("base", "--pointer-loss-weight", "1.0")
        with self.assertRaises(SystemExit):
            launcher.parse_args([
                "--mode", "t3b", "--model-size", "base",
                "--data-root", "/data", "--output-dir", "/outputs",
                "--pointer-loss-weight", "nan",
            ])

    def test_balanced_variable_loss_requires_supervision_and_is_forwarded(self):
        args = self.args(
            "base", "--variable-loss-weight", "0.1", "--balanced-variable-loss"
        )
        self.assertIn("--balanced-variable-loss", launcher.build_command(args))
        with self.assertRaises(SystemExit):
            self.args("base", "--balanced-variable-loss")

    def test_transition_loss_requires_balancing_and_is_forwarded(self):
        args = self.args(
            "base", "--variable-loss-weight", "1.0",
            "--balanced-variable-loss", "--transition-variable-loss",
        )
        self.assertIn("--transition-variable-loss", launcher.build_command(args))
        with self.assertRaises(SystemExit):
            self.args("base", "--variable-loss-weight", "1.0", "--transition-variable-loss")

    def test_causal_readout_requires_transition_and_is_forwarded(self):
        args = self.args(
            "base", "--variable-loss-weight", "1.0", "--balanced-variable-loss",
            "--transition-variable-loss", "--causal-task-readout",
        )
        self.assertIn("--causal-task-readout", launcher.build_command(args))
        with self.assertRaises(SystemExit):
            self.args("base", "--variable-loss-weight", "1.0", "--causal-task-readout")

    def test_hard_pointer_requires_causal_t3b_and_is_forwarded(self):
        args = launcher.parse_args([
            "--mode", "t3b", "--model-size", "base", "--data-root", "/data",
            "--output-dir", "/outputs", "--variable-loss-weight", "1.0",
            "--balanced-variable-loss", "--transition-variable-loss",
            "--causal-task-readout", "--hard-pointer",
        ])
        self.assertIn("--hard-pointer", launcher.build_command(args))
        with self.assertRaises(SystemExit):
            self.args("base", "--hard-pointer")

    def test_optional_paraphrase_catalog_is_forwarded(self):
        args = self.args("base", "--paraphrase-catalog", "/catalog.json")
        command = launcher.build_command(args)
        index = command.index("--paraphrase-catalog")
        self.assertEqual(command[index + 1], "/catalog.json")

    def test_source_filter_is_forwarded(self):
        command = launcher.build_command(self.args("base", "--sources", "ccrgb"))
        self.assertEqual(command[-2:], ["--sources", "ccrgb"])

    def test_no_test_or_a1_claim_flags_exist(self):
        option_strings = {
            option for action in launcher.parse_args.__globals__["argparse"].ArgumentParser()._actions
            for option in action.option_strings
        }
        command = launcher.build_command(self.args())
        self.assertNotIn("--test", command)
        self.assertNotIn("--include-test", command)
        self.assertNotIn("--a1", command)

    def test_evidence_capable_requires_and_forwards_group_sidecars(self):
        with self.assertRaises(SystemExit):
            self.args("base", "--evidence-capable")
        args = self.args(
            "base", "--groups-dir", "/groups", "--evidence-capable"
        )
        command = launcher.build_command(args)
        self.assertIn("--groups-dir", command)
        self.assertIn("--evidence-capable", command)

    def test_authoritative_xor_bundles_are_forwarded_with_intrinsic_groups(self):
        args = launcher.parse_args([
            "--mode", "t3b", "--model-size", "base", "--data-root", "/data",
            "--output-dir", "/outputs", "--sources", "xor",
            "--xor-bundle", "/graphs/3001", "/pairs/3001.jsonl",
            "--evidence-capable",
        ])
        command = launcher.build_command(args)
        index = command.index("--xor-bundle")
        self.assertEqual(
            command[index + 1:index + 3],
            ["/graphs/3001", "/pairs/3001.jsonl"],
        )
        with self.assertRaises(SystemExit):
            self.args(
                "base", "--sources", "wiqa", "--xor-bundle",
                "/graphs/3001", "/pairs/3001.jsonl",
            )

    def test_trainer_uses_scientific_reader_and_stays_non_a1(self):
        source = (Path(__file__).parents[1] / "src" / "train_addressed.py").read_text()
        self.assertIn("ScientificAddressedReader", source)
        self.assertIn("tokenize_scientific_fields", source)
        self.assertIn('"a1_evidence": False', source)
        self.assertIn("record.probes[].answer_after only", source)
        self.assertIn("variable_cross_entropy", source)
        self.assertIn("load_paraphrase_catalog", source)
        self.assertIn('"unseen_paraphrase"', source)
        self.assertNotIn("append_factual_question", source)


if __name__ == "__main__":
    unittest.main()
