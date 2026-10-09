# CORE benchmark data

This repository converts public causal-reasoning datasets into the shared CORE intervention-record format.

The current phase is CPU-only data acquisition, parsing, validation, documentation, and split construction. Raw third-party datasets stay under `data/raw/` and are not committed.

## Current status

The four priority sources are complete. Com2 contributes 770 validated records, CLadder contributes 1,422 deterministic counterfactual DAG records, WIQA contributes 26,045 signed direction-of-change DAG records, and CCR.GB contributes 36,000 generated DAG records from 6,000 independent worlds. All six lower-priority sources have also been pinned, downloaded, inspected, and passed through source-specific converters; none has enough paired intervention evidence for this fixed schema, so their candidates are explicitly quarantined. All 64,237 accepted records have complete split coverage.

Run the repository readiness checks with:

```bash
make check
```

Run verification against generated benchmark records with:

```bash
make verify
```

Regenerate Com2 records and its audit card with:

```bash
make convert-com2
make document-com2
```

Regenerate CLadder records and its audit card with:

```bash
make convert-cladder
make document-cladder
```

Regenerate WIQA records and its audit card with:

```bash
make convert-wiqa
make document-wiqa
```

Regenerate CCR.GB records and its audit card from the external 6,000-world raw output with:

```bash
make convert-ccrgb CCRGB_RAW=/absolute/path/to/ccrgb_6000_worlds.jsonl
make document-ccrgb CCRGB_RAW=/absolute/path/to/ccrgb_6000_worlds.jsonl
```

Regenerate all lower-priority contract assessments and cards with:

```bash
make convert-lower
make document-lower
```

See `docs/WORK_PLAN.md` for the source order and the gate before model experiments.
