# Evidence Inference 2.0 source card for G4

Status: development source acquired and hash-pinned; comparative-evidence adapter pending.

## Source and integrity

- Repository revision: `not-published`
- Source: `https://github.com/jayded/evidence-inference`
- Local raw snapshot: `data/raw/evidence_inference/evidence-inference-a661e8c1-development-source.tar.gz`
- SHA-256: `not-published`
- Licence: repository `LICENSE`, preserved in the raw snapshot

The snapshot includes prompts, Evidence Inference 2.0 annotations, abstracts, and official split files. Development-only inspection finds 743 train articles, 92 validation articles, 2,412 development prompts, 4,366 annotation rows, and 4,077 rows whose labels passed source validation.

## Task boundary

The source directly identifies intervention, comparator, outcome, direction label, and supporting evidence for randomized trials. It is therefore suitable for a text-native biomedical intervention measurement. It is not a complete DAG/world-state corpus and must not be presented as one.

G4 will use an explicit comparative-evidence amendment: baseline versus O3 on intervention/comparator/outcome direction, grouped at article and prompt level. PubMedCausal remains auxiliary span-extraction data because the 300-item annotation sample yielded zero executable intervention pairs.

## Test isolation

`src/inspect_evidence_inference.py` reads only the official Evidence Inference 2.0 train and validation article-ID files. Any request for the official test split fails closed. The held-out test is not evaluated.

The next step is to normalize source-validated prompt/evidence rows, collapse duplicate doctor annotations under a frozen agreement rule, preserve article groups, and write train/validation manifests.
