"""Render data cards for lower-priority sources after contract assessment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from pathlib import Path


SEED = 20260903
CONFIG = {
    "counterbench": {
        "title": "CounterBench",
        "url": "https://huggingface.co/datasets/CounterBench/CounterBench/tree/not-published",
        "license": "`license: mit` (verbatim dataset-card metadata)",
        "mapping": "The natural-language causal clauses can suggest graph arcs, but the release supplies only the counterfactual answer. It does not supply a factual outcome/state pair. Joint and nested rows also contain multiple actions while CORE has one intervention target.",
    },
    "causalt5k": {
        "title": "CausalT5K",
        "url": "https://github.com/genglongling/CausalT5kBench/tree/not-published",
        "license": "Dataset: `CC-BY-4.0`; code: `MIT` (verbatim identifiers from the repository README).",
        "mapping": "The ten named domains and Pearl levels are preserved in quarantine metadata. `variables` and free-text `causal_structure` do not constitute a complete machine-readable edge list, and no paired world states are released.",
    },
    "corr2cause": {
        "title": "Corr2Cause paraphrased test",
        "url": "https://huggingface.co/datasets/causal-nlp/corr2cause/tree/not-published",
        "license": "none stated in the pinned dataset card or downloaded files",
        "mapping": "Premise/hypothesis/relation classification tests whether correlations identify a causal relation. It has no explicit action and no before/after world pair.",
    },
    "crass": {
        "title": "CRASS",
        "url": "https://github.com/apergo-ai/CRASS-data-set/tree/not-published",
        "license": "`The CRASS benchmark data set is released under the Apache License 2.0.` (verbatim README sentence)",
        "mapping": "Premise, counterfactual question, and correct option are retained in quarantine. There is no explicit graph, complete factual state, or intervened state.",
    },
    "meter": {
        "title": "METER",
        "url": "https://github.com/SCUNLP/METER/tree/not-published",
        "license": "none stated; the pinned repository contains no licence file and its README is only `# METER`",
        "mapping": "Each context's causal-discovery, intervention, and counterfactual multiple-choice questions are enumerated. None includes an explicit graph plus paired factual/intervened states.",
    },
    "pubmedcausal": {
        "title": "PubMedCausal",
        "url": "https://huggingface.co/datasets/jaypee01/PubMedCausal/tree/not-published",
        "license": "MIT License; `Copyright (c) 2026 edyahlimited` (verbatim opening lines)",
        "mapping": "The 16 wide Cause/Effect groups are reshaped into one long candidate per populated relation. Rows with no relation remain one negative candidate. These are span relations, not interventions or paired world states.",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def raw_example(source: str, raw_root: Path) -> object:
    if source == "counterbench":
        return json.loads((raw_root / source / "data_balanced_alpha_V1.json").read_text(encoding="utf-8"))[0]
    if source == "corr2cause":
        return json.loads((raw_root / source / "perturbation_by_paraphrasing_test.json").read_text(encoding="utf-8"))[0]
    if source == "pubmedcausal":
        return json.loads((raw_root / source / "30k_train.json").read_text(encoding="utf-8"))[0]
    tar_path = next((raw_root / source).glob("*.tar.gz"))
    with tarfile.open(tar_path, "r:gz") as archive:
        if source == "causalt5k":
            member = next(m for m in archive.getmembers() if m.name.endswith("/final_dataset/D1/D1_L1.json"))
            return json.loads(archive.extractfile(member).read().decode("utf-8"))[0]
        if source == "crass":
            member = next(m for m in archive.getmembers() if m.name.endswith("/CRASS_FTM_main_data_set.csv"))
            text = archive.extractfile(member).read().decode("utf-8-sig")
            return next(csv.DictReader(io.StringIO(text), delimiter=";"))
        member = next(m for m in archive.getmembers() if m.name.endswith("/dataset.jsonl"))
        first = archive.extractfile(member).readline().decode("utf-8")
        return json.loads(first)


def fenced(value: object) -> str:
    return "~~~json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n~~~"


def render(source: str, raw_root: Path, records_dir: Path, reports_dir: Path) -> str:
    cfg = CONFIG[source]
    summary = json.loads((reports_dir / f"{source}_conversion_summary.json").read_text(encoding="utf-8"))
    rejected = read_jsonl(records_dir / f"{source}.rejected.jsonl")
    samples = random.Random(SEED).sample(rejected, min(5, len(rejected)))
    parts = [
        f"# {cfg['title']} data card",
        "",
        "Status: raw source acquired and fully assessed; zero CORE records accepted because the release cannot satisfy the fixed paired-intervention contract without invented data.",
        "",
        "## Source and licence",
        "",
        f"- Pinned source: {cfg['url']}",
        f"- Licence: {cfg['license']}",
        f"- Raw inventory/hashes: `{json.dumps(summary.get('raw_files', {'archive_sha256': summary.get('raw_sha256')}), sort_keys=True)}`",
        "",
        "## Counts and decision",
        "",
        f"- Raw/candidate questions: {summary.get('raw_questions', summary.get('long_candidates')):,}",
        f"- Accepted: {summary['accepted']:,}",
        f"- Quarantined: {summary['rejected']:,}",
        "",
        cfg["mapping"],
        "",
        "No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.",
        "",
        "## Full raw example",
        "",
        fenced(raw_example(source, raw_root)),
        "",
        "## Reproducible quarantine spot checks",
        "",
        f"There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random({SEED})`; each retains its exact rejection reason so the zero-record decision is auditable.",
    ]
    for index, sample in enumerate(samples, 1):
        parts += ["", f"### Candidate {index}", "", fenced(sample)]
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(CONFIG))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    sources = [args.source] if args.source else sorted(CONFIG)
    for source in sources:
        output = args.docs_dir / f"{source.upper()}_CARD.md"
        output.write_text(render(source, args.raw_root, args.records_dir, args.reports_dir), encoding="utf-8")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
