#!/usr/bin/env python3
"""Independent adjudication of frozen blinded WIQA and Com2 packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STRUCTURAL = (
    "gold_final_outputs", "target_variables", "actually_changed_variables",
    "preserved_variables", "first_edit_valid", "second_edit_valid",
)
COM2_INTERMEDIATE_CHOICE = {
    "manual-two-edit:com2:3104da8494623f192692345e": "proposal_A",
    "manual-two-edit:com2:50e02d94f918b6cefe1311b6": "proposal_B",
    "manual-two-edit:com2:5d398cf082a20777fffeb45a": "proposal_B",
    "manual-two-edit:com2:944a0ed0d07748e1dc2b9fab": "proposal_B",
    "manual-two-edit:com2:a8eedb88900dcca8a252036e": "proposal_B",
    "manual-two-edit:com2:e223c9b1cda2cf01c0664fd4": "proposal_A",
    "manual-two-edit:com2:ebb15cfb1f9ff31e3855eb6f": "proposal_A",
}


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalpha())


def queried_wiqa_node(item: dict[str, Any]) -> str:
    question = item["factual_question"]
    marker = "how will it affect"
    if marker not in question.lower():
        raise ValueError("WIQA question lacks outcome marker")
    outcome = normalized(question.lower().split(marker, 1)[1])
    matches = {
        node for node, surfaces in item["graph"]["node_text"].items()
        if any(normalized(surface) == outcome for surface in surfaces)
    }
    if len(matches) != 1:
        raise ValueError(f"WIQA outcome must ground to exactly one node, found {sorted(matches)}")
    return next(iter(matches))


def base_output(row: dict[str, Any], selected: dict[str, Any], decision: str, justification: str) -> dict[str, Any]:
    return {
        "pair_id": row["pair_id"], "annotator_id": "Independent Reviewer",
        "source": row["source"], "split": "validation",
        "intermediate_outputs": selected["intermediate_outputs"],
        "gold_final_outputs": selected["gold_final_outputs"],
        "query_answer_after_both_edits": selected["query_answer_after_both_edits"],
        "target_variables": selected["target_variables"],
        "actually_changed_variables": selected["actually_changed_variables"],
        "preserved_variables": selected["preserved_variables"],
        "first_edit_valid": selected["first_edit_valid"],
        "second_edit_valid": selected["second_edit_valid"],
        "status": "accepted", "confidence": "high", "justification": justification,
        "adjudication_decision": decision,
    }


def adjudicate_wiqa(row: dict[str, Any]) -> dict[str, Any]:
    first, second = row["proposal_A"], row["proposal_B"]
    if any(first[field] != second[field] for field in STRUCTURAL):
        raise ValueError(f"{row['pair_id']}: WIQA structural proposals unexpectedly differ")
    node = queried_wiqa_node(row["annotation_item"])
    answer = first["gold_final_outputs"][node]
    if answer not in {"more", "less", "no_effect"}:
        raise ValueError(f"{row['pair_id']}: queried final value is outside vocabulary")
    candidates = [name for name in ("proposal_A", "proposal_B") if row[name]["query_answer_after_both_edits"] == answer]
    if len(candidates) != 1:
        raise ValueError(f"{row['pair_id']}: exactly one proposal must copy queried final value")
    choice = candidates[0]
    selected = dict(row[choice])
    selected["query_answer_after_both_edits"] = answer
    surface = next(
        text for text in row["annotation_item"]["graph"]["node_text"][node]
        if normalized(text) == normalized(row["annotation_item"]["factual_question"].lower().split("how will it affect", 1)[1])
    )
    return base_output(
        row, selected, choice[-1],
        f"The outcome phrase {surface!r} grounds exactly to node {node}. Both blinded proposals agree that its final value is {answer}; the query label must exactly copy that value.",
    )


def event_index(target: str) -> int:
    return int(target.split(":", 1)[0].split("_", 1)[1])


def adjudicate_com2(row: dict[str, Any]) -> dict[str, Any]:
    pair_id = row["pair_id"]
    if pair_id not in COM2_INTERMEDIATE_CHOICE:
        raise ValueError(f"unexpected Com2 adjudication row {pair_id}")
    item = row["annotation_item"]
    chain = item["chain"]
    first_index = event_index(item["first_edit"]["target"])
    second_index = event_index(item["second_edit"]["target"])
    if second_index != first_index + 1:
        raise ValueError("Com2 adjudication expects the frozen adjacent-edit protocol")
    final = {f"event_{index:02d}": value for index, value in enumerate(chain)}
    final[f"event_{first_index:02d}"] = item["first_edit"]["value"]
    final[f"event_{second_index:02d}"] = item["second_edit"]["value"]
    variables = list(final)
    changed = [name for index, name in enumerate(variables) if final[name] != chain[index]]
    preserved = [name for name in variables if name not in changed]
    targets = list(dict.fromkeys((f"event_{first_index:02d}", f"event_{second_index:02d}")))
    chosen = row[COM2_INTERMEDIATE_CHOICE[pair_id]]
    selected = {
        **chosen, "gold_final_outputs": final,
        "query_answer_after_both_edits": final[variables[-1]],
        "target_variables": targets, "actually_changed_variables": changed,
        "preserved_variables": preserved, "first_edit_valid": True, "second_edit_valid": True,
    }
    return base_output(
        row, selected, "custom",
        "The first replacement remains fixed at its addressed chain position. The adjacent second intervention severs the incoming link at that position and restores its supplied factual event; the unchanged factual continuation then follows downstream. Final strings are canonicalized exactly from the frozen intervention values and factual chain rather than choosing between stylistic variants.",
    )


def adjudicate(packet: Path, output: Path, source: str) -> dict[str, Any]:
    rows = read(packet)
    if not rows or any(row.get("source") != source or row.get("split") != "validation" for row in rows):
        raise ValueError("packet source/split mismatch")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("packet contains duplicate pair IDs")
    function = adjudicate_wiqa if source == "wiqa" else adjudicate_com2
    decisions = [function(row) for row in rows]
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    counts = {name: sum(row["adjudication_decision"] == name for row in decisions) for name in ("A", "B", "custom")}
    return {
        "source": source, "rows": len(decisions), "decisions": counts,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "adjudicator_id": "Independent Reviewer", "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("wiqa", "com2"), required=True)
    args = parser.parse_args()
    result = adjudicate(args.packet, args.output, args.source)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
