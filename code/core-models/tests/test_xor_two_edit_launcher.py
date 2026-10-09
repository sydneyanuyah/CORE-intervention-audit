import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts" / "launch_xor_two_edit_eval.py"
SPEC = importlib.util.spec_from_file_location("xor_two_edit_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class XORTwoEditLauncherTests(unittest.TestCase):
    def args(self, *extra):
        return launcher.parse_args([
            "--bundle", "/artifacts/graph-1", "/manifests/graph-1.jsonl",
            "--checkpoint", "/checkpoints/best.pt",
            "--output-json", "/outputs/validation.json",
            "--mode", "t3b",
            *extra,
        ])

    def test_launches_exactly_four_processes_and_forwards_bundle(self):
        command = launcher.build_command(self.args())
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(command.count("--bundle"), 1)
        index = command.index("--bundle")
        self.assertEqual(
            command[index + 1 : index + 3],
            ["/artifacts/graph-1", "/manifests/graph-1.jsonl"],
        )

    def test_launcher_hard_blocks_test(self):
        with self.assertRaises(SystemExit):
            self.args("--split", "test")

    def test_forwards_paraphrase_catalog(self):
        command = launcher.build_command(
            self.args("--family", "wiqa", "--paraphrase-catalog", "/catalog.json")
        )
        index = command.index("--paraphrase-catalog")
        self.assertEqual(command[index + 1], "/catalog.json")

    def test_forwards_validation_only_a2_control(self):
        args = launcher.parse_args([
            "--bundle", "/artifacts/graph-1", "/manifests/graph-1.jsonl",
            "--checkpoint", "/checkpoints/best.pt",
            "--output-json", "/outputs/validation.json",
            "--mode", "t2b",
            "--a2-control", "no_instruction",
        ])
        command = launcher.build_command(args)
        index = command.index("--a2-control")
        self.assertEqual(command[index + 1], "no_instruction")

    def test_rejects_a2_control_for_non_t2b(self):
        with self.assertRaises(SystemExit):
            self.args("--a2-control", "editor_zeroed")


if __name__ == "__main__":
    unittest.main()
