import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_cladder import parse_structure  # noqa: E402


class CLadderInspectionTests(unittest.TestCase):
    def test_structure_parser_preserves_edges(self):
        nodes, edges = parse_structure("X->V2,X->Y,V2->Y")
        self.assertEqual(nodes, ["V2", "X", "Y"])
        self.assertEqual(edges, [["X", "V2"], ["X", "Y"], ["V2", "Y"]])


if __name__ == "__main__":
    unittest.main()
