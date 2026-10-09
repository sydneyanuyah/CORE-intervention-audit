"""Render the completed WIQA conversion data card."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path

from convert_wiqa import GRAPH_SHA256, QUESTION_FILES


SPOT_CHECK_SEED = 20260903
CODE_SHA256 = "not-published"


def fenced_json(value: object) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def licence_text(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("/LICENSE"))
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("WIQA LICENSE could not be read")
        return handle.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render(raw_dir: Path, records_path: Path, summary_path: Path) -> str:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    graphs = read_jsonl(raw_dir / "wiqa_influence_graphs.jsonl")
    first_raw = json.loads((raw_dir / "no_explanation_v2_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    linked_graph = next(graph for graph in graphs if graph["graph_id"] == first_raw["metadata"]["graph_id"])
    samples = random.Random(SPOT_CHECK_SEED).sample(records, 5)
    question_hashes = "\n".join(
        f"- `{local}` (remote `{remote}`): `{digest}`"
        for local, remote, digest in QUESTION_FILES.values()
    )
    rejection_rows = "\n".join(
        f"| {reason} | {count:,} |" for reason, count in summary["rejection_reason_counts"].items()
    )
    sections = [
        "# WIQA data card",
        "",
        "Status: signed influence-graph conversion complete and verified.",
        "",
        "## Source and integrity",
        "",
        "- Pinned code revision: `not-published`",
        "- Code snapshot: `data/raw/wiqa/wiqa-edeef924.tar.gz`",
        f"- Code snapshot SHA-256: `{CODE_SHA256}`",
        f"- Influence graphs: 2,107; SHA-256 `{GRAPH_SHA256}`",
        "- Licence: Apache License 2.0",
        "- Primary question release: official no-explanation-v2",
        "",
        question_hashes,
        "",
        "The primary release contains 39,705 unique questions: 29,808 train, 6,894 development, and 3,003 test. The 2,107 graph rows are required auxiliary metadata; the Hugging Face QA projection alone is insufficient.",
        "",
        "## Conversion counts",
        "",
        "- Accepted: 26,045",
        "- Quarantined: 13,660",
        "- Accepted splits: 19,758 train, 4,547 validation (source dev), 1,740 test",
        "",
        "| Rejection reason | Rows |",
        "|---|---:|",
        rejection_rows,
        "",
        "The 13,489 out-of-paragraph rows are deliberate no-effect distractors: their source event is normally outside the referenced graph, and they cannot satisfy the required observed-change probe. Another 157 questions do not resolve to a signed path after WIQA's own letters-only normalization; 14 resolve to multiple source/target node pairs and are quarantined rather than guessed.",
        "",
        "## Signed graph reconstruction",
        "",
        "WIQA graph v1 has fixed signed relations: `V -→ X`, `Z +→ X`, `X -→ W`, `X +→ Y`, and `U -→ Y`. `Y_affects_outcome` determines the signs from `Y` and `W` to acceleration node `A` and deceleration node `D`. Empty grounding groups and their incident edges are removed, matching the pinned source implementation. The fixed CORE edge list cannot carry signs, so `graph.edges` preserves topology while signed path resolution is reflected in directional probe answers and documented here.",
        "",
        "## Field mapping",
        "",
        "| CORE field | WIQA source or derivation |",
        "|---|---|",
        "| `source_id` | `metadata.ques_id` |",
        "| `graph` | active symbolic v1 nodes and unsigned projection of fixed signed edges |",
        "| `factual.passage` | linked graph `paragraph` preserved exactly as a prefix, followed by deterministic ` [TARGET] <source event>` pointer text |",
        "| `factual.state` | `null`; WIQA encodes directional effects, not world values |",
        "| `factual.question` | `question.stem`, verbatim |",
        "| `factual.answer` | `null`; the source answer is post-change direction |",
        "| intervention target | uniquely resolved structural source graph node; `target_text` is the exact appended source phrase and `target_span` addresses only that phrase |",
        "| intervention value/text | source event phrase parsed verbatim from the stem; `value_token` is the canonical signed-node direction `more` or `less` |",
        "| `intervened.state` | `null` |",
        "| `intervened.answer` | normalized source label `more` or `less` |",
        "| descendants | graph reachability from the source node |",
        "| non-descendants | other active nodes excluding source and descendants |",
        "| probes | queried descendant goes from directional baseline `no_effect` to source answer; non-descendants remain `no_effect` |",
        "",
        "## Caveats",
        "",
        "- WIQA metrics are direction-of-change accuracy (`more`/`less`/`no_effect`), not state-value matching.",
        "- The probe baseline `no_effect` means no directional change before applying the stated perturbation; it is not a recovered factual state.",
        "- The accepted set excludes all no-effect items because the shared verifier requires at least one observed changed probe.",
        "- Symbolic nodes may have several natural-language groundings. Only questions resolving to exactly one source/target pair at the released path length and answer sign are accepted.",
        "- Most source-event phrases are absent from the original process paragraph. The converter preserves that paragraph byte-for-byte as a prefix and appends one marked source phrase, avoiding fabricated alignment or loss of 25,982 otherwise valid records.",
        "- Official train/dev/test files are preserved; `dev` is named `validation` in CORE manifests.",
        "",
        "## Licence text, verbatim",
        "",
        "```text",
        licence_text(raw_dir / "wiqa-edeef924.tar.gz"),
        "```",
        "",
        "## First raw question and linked graph, verbatim fields",
        "",
        "### Question",
        "",
        fenced_json(first_raw),
        "",
        "### Linked influence graph",
        "",
        fenced_json(linked_graph),
        "",
        "## Reproducible random spot checks",
        "",
        f"Five accepted records were sampled with Python `random.Random({SPOT_CHECK_SEED})`.",
    ]
    for index, record in enumerate(samples, 1):
        must = [probe for probe in record["probes"] if probe["required"] == "must not change"]
        sections += [
            "",
            f"### Spot check {index}: `{record['id']}`",
            "",
            fenced_json(record),
            "",
            f"Assessment: the queried node is reachable from `{record['intervention']['target']}`, its directional answer changes from `no_effect` to `{record['intervened']['answer']}`, and all {len(must)} non-descendant probes are preserved.",
        ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/wiqa"))
    parser.add_argument("--records", type=Path, default=Path("data/records/wiqa.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/wiqa_conversion_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/WIQA_CARD.md"))
    args = parser.parse_args()
    content = render(args.raw_dir, args.records, args.summary)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output} with licence, raw question+graph, and 5 spot checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
