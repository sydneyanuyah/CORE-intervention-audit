import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.xor_two_edit import generate_xor_two_edit_manifest, xor_scm_state


def fixture():
    nodes = [f"X{i:02d}" for i in range(8)]
    graph = {
        "nodes": nodes,
        "root_nodes": nodes[:2],
        "parents": {**{node: [] for node in nodes[:2]}, **{node: [nodes[i - 1]] for i, node in enumerate(nodes[2:], 2)}},
        "xor_bias": {node: 0 for node in nodes},
        "seed": 42,
    }
    roots = {"X00": 0, "X01": 1}
    factual = xor_scm_state(graph, roots)
    world = {"world_id": "world-00", "split": "train", "root_values": roots, "factual_state": factual}
    records = []
    for target in nodes:
        for value in (0, 1):
            records.append({
                "record_id": f"world-00-{target}-{value}",
                "world_id": "world-00",
                "intervention": {"target": target, "value": value},
                "intervened_state": xor_scm_state(graph, roots, {target: value}),
            })
    return graph, [world], records


class XorTwoEditTests(unittest.TestCase):
    def test_truth_is_recomputed_and_schedule_is_deterministic(self):
        graph, worlds, records = fixture()
        first = generate_xor_two_edit_manifest(
            graph, worlds, records, graph_group_id="xor:graph:42"
        )
        second = generate_xor_two_edit_manifest(
            graph, list(reversed(worlds)), list(reversed(records)), graph_group_id="xor:graph:42"
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(sum(row["target_mask"]) == 2 for row in first))
        self.assertTrue(all(row["truth_provenance"]["a1_evidence"] is False for row in first))

    def test_tampered_single_edit_truth_fails_closed(self):
        graph, worlds, records = fixture()
        records[1]["intervened_state"]["X07"] ^= 1
        with self.assertRaisesRegex(ValueError, "single-edit truth disagrees"):
            generate_xor_two_edit_manifest(
                graph, worlds, records, graph_group_id="xor:graph:42"
            )

    def test_tampered_factual_truth_fails_closed(self):
        graph, worlds, records = fixture()
        worlds[0]["factual_state"]["X07"] ^= 1
        with self.assertRaisesRegex(ValueError, "factual state disagrees"):
            generate_xor_two_edit_manifest(
                graph, worlds, records, graph_group_id="xor:graph:42"
            )


if __name__ == "__main__":
    unittest.main()
