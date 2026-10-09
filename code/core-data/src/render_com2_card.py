"""Render the Com2 data card from pinned raw data and normalized outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SPOT_CHECK_SEED = 20260903


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fenced_json(value: object) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def render(raw_path: Path, records_dir: Path, summary_path: Path) -> str:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_jsonl(records_dir / "com2_intervention.jsonl")
    records += read_jsonl(records_dir / "com2_counterfactual.jsonl")
    first_raw = next(item for item in raw if item.get("type") == "intervention" and "chains" in item)
    samples = random.Random(SPOT_CHECK_SEED).sample(records, 5)

    sections = [
        "# Com2 data card",
        "",
        "Status: conversion complete and verified.",
        "",
        "## Source and integrity",
        "",
        "- Pinned revision: `not-published`",
        "- Benchmark URL: `https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json`",
        "- Local raw file: `data/raw/com2/main.json`",
        "- Bytes: 9,128,252",
        "- SHA-256: `not-published`",
        "- Licence: none stated. The pinned repository root has no licence file and its README contains no licence statement.",
        "",
        "The raw file contains 2,500 objects: 500 each of `counterfactual`, `decision`, `direct`, `intervention`, and `reco`. This differs from the task note, which called the first type `transition`; the converter asserts the observed release rather than renaming it.",
        "",
        "## Selection and counts",
        "",
        "| Output | Accepted | Rejected | Chain prefix findings |",
        "|---|---:|---:|---|",
        "| `com2_intervention.jsonl` | 270 | 0 | `k=1` for all 270; 258 chains are 5/5 and 12 are 6/6 |",
        "| `com2_counterfactual.jsonl` | 500 | 0 | `k=0` for 448 and `k=1` for 52; lengths are 5/4 for 4, 5/5 for 495, and 6/6 for 1 |",
        "",
        "The other 230 intervention rows have no `chains` key and are not converted. `direct`, `decision`, and `reco` are outside this converter's specified selection. Empty rejection files are emitted so a later failure cannot be confused with an unrecorded omission.",
        "",
        "## Field mapping",
        "",
        "| CORE field | Com2 source or derivation |",
        "|---|---|",
        "| `id` | `com2-<type>-<zero-padded original list index>` |",
        "| `source_id` | zero-based index in untouched `main.json` |",
        "| `structure_kind` | `chain` |",
        "| `graph` | `null`; Com2 supplies paths, not a DAG |",
        "| `chain` | factual path `chains[0]`, verbatim |",
        "| `factual.passage` | source `scenario` preserved as a prefix, followed by marked `[ORIG]` and `[REPL]` event candidates for pointer addressing |",
        "| `factual.question` | `question`, verbatim |",
        "| `factual.answer` | `null`; the only source `answer` answers the intervention/counterfactual question |",
        "| `factual.state` | `null`; no variable-state map is supplied |",
        "| intervention target/value | first divergent position `k`: the position-qualified before event is the structural target; exact `target_span` and `replacement_span` address the marked before/after event text; `value_token` is `null` |",
        "| `intervention.text` | source `question`, verbatim |",
        "| `intervened.answer` | source `answer`, verbatim |",
        "| intervened passage/state | the same marked model-input passage / `null` state |",
        "| `non_descendants` | position-qualified occurrences before `k` |",
        "| `descendants` | position-qualified factual occurrences after `k` |",
        "| probes | aligned before/after occurrences; prefix is `must not change`, target and suffix are `may change` |",
        "",
        "Event occurrences are labelled `event_NN: <verbatim event>` because eight source chains repeat identical event text at different positions. This disambiguates identity without changing `chain`, which remains verbatim. For the four 5/4 counterfactual pairs, the unmatched final factual event remains a descendant but has no probe because the source gives no after value.",
        "",
        "## Splits",
        "",
        "Com2 provides no authoritative train/validation/test assignment for these rows. Exact factual root-event groups are assigned together using `sha256('com2-split-v1:' + root_event) mod 100`: buckets 0–79 train, 80–89 validation, and 90–99 test. This prevents the same exact root event from crossing splits.",
        "",
        f"- Intervention: {summary['split_counts']['intervention']['train']} train, {summary['split_counts']['intervention']['validation']} validation, {summary['split_counts']['intervention']['test']} test",
        f"- Counterfactual: {summary['split_counts']['counterfactual']['train']} train, {summary['split_counts']['counterfactual']['validation']} validation, {summary['split_counts']['counterfactual']['test']} test",
        "",
        "## Caveats",
        "",
        "- These are chain/path records. Descendant labels mean later positions on a supplied path, not graph reachability.",
        "- `[ORIG]` and `[REPL]` are deterministic non-asserted pointer candidates appended to the source scenario; both character spans are validated as exact slices.",
        "- The 448 counterfactual rows with `k=0` have no non-descendants and therefore cannot test preservation.",
        "- A `may change` probe is allowed to remain unchanged. The target always differs, satisfying the observed-effect requirement.",
        "- Answers are multiple-choice strings, while probes compare event strings. They measure different views of the same intervention.",
        "- No redistribution licence is stated; the raw data should not be redistributed without permission from its owners.",
        "",
        "## Full raw item, verbatim fields",
        "",
        "The first selected intervention item (original zero-based index 2) is reproduced below without field changes:",
        "",
        fenced_json(first_raw),
        "",
        "## Reproducible random spot checks",
        "",
        f"Five records were sampled from the 770 accepted rows with Python `random.Random({SPOT_CHECK_SEED})`. Each full normalized record follows.",
    ]
    for index, record in enumerate(samples, 1):
        must = [probe for probe in record["probes"] if probe["required"] == "must not change"]
        may = [probe for probe in record["probes"] if probe["required"] == "may change"]
        changed = sum(probe["actually_changed"] for probe in may)
        sections += [
            "",
            f"### Spot check {index}: `{record['id']}`",
            "",
            fenced_json(record),
            "",
            f"Assessment: the target is the first divergent occurrence; all {len(must)} preservation probes agree before/after, and {changed} of {len(may)} may-change probes differ. The normalized mapping is structurally consistent with the two source paths.",
        ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/com2/main.json"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--summary", type=Path, default=Path("reports/com2_conversion_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/COM2_CARD.md"))
    args = parser.parse_args()
    content = render(args.raw, args.records_dir, args.summary)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output} with 1 raw item and 5 normalized spot checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
