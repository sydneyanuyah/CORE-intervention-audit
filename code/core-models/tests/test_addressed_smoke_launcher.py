import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts" / "launch_addressed_smoke.py"
SPEC = importlib.util.spec_from_file_location("addressed_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class AddressedSmokeLauncherTests(unittest.TestCase):
    def parse(self, *extra):
        return launcher.parse_args([
            "--mode", "t2b", "--data-root", "/data", "--output-json", "/tmp/run.json",
            *extra,
        ])

    def test_exact_four_gpu_torchrun_policy_and_defaults(self):
        args = self.parse()
        command = launcher.build_command(args)
        self.assertEqual(launcher.GPU_COUNT, 4)
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(args.max_length, 512)
        self.assertEqual(args.steps, 2)

    def test_all_modes_are_registered(self):
        self.assertEqual(launcher.MODES, ("t2a", "t2b", "t3a", "t3b"))

    def test_a2_editor_disabled_is_forwarded(self):
        command = launcher.build_command(self.parse("--editor-disabled"))
        self.assertIn("--editor-disabled", command)

    def test_a2_rejected_for_t3(self):
        with self.assertRaises(SystemExit):
            launcher.parse_args([
                "--mode", "t3b", "--data-root", "/data", "--output-json", "/tmp/x.json",
                "--editor-disabled",
            ])

    def test_a3_override_rejected_for_t2_and_forwarded_for_t3(self):
        with self.assertRaises(SystemExit):
            self.parse("--span-override", "random")
        args = launcher.parse_args([
            "--mode", "t3b", "--data-root", "/data", "--output-json", "/tmp/x.json",
            "--span-override", "adjacent",
        ])
        self.assertIn("adjacent", launcher.build_command(args))


if __name__ == "__main__":
    unittest.main()
