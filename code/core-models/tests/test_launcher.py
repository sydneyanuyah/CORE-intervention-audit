import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts" / "launch_operator_matrix.py"
SPEC = importlib.util.spec_from_file_location("launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_gpu_policies(self):
        self.assertEqual(launcher.MODELS["base"]["gpus"], 4)
        self.assertEqual(launcher.MODELS["large"]["gpus"], 8)

    def test_integer_lists(self):
        self.assertEqual(launcher.parse_ints("0,2,7"), [0, 2, 7])
