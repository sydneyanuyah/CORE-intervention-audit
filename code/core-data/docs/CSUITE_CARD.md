# CSuite source card for T2/T4

Status: all 15 version 0.1 archives acquired and hash-pinned; development adapter pending.

## Source and integrity

- Repository revision: `not-published`
- Release: `v0.1`
- Source: `https://github.com/microsoft/csuite`
- Local raw directory: `data/raw/csuite/`
- Archive count: 15
- Total compressed size: approximately 28 MiB
- Per-archive SHA-256 values: `reports/csuite_schema_summary.json`

The release supplies a true adjacency matrix plus observational train and validation files for every named SEM. Those 15 named SEMs provide stable domain identities for a held-out-family design.

## Test isolation

The release names `test.csv`, `interventions.json`, and (for some families) `counterfactuals.json` as test artifacts. They are sealed. `src/inspect_csuite.py` refuses to read those members and reports `test_evaluated=false`.

T2/T4 development interventions must be regenerated from the pinned official CSuite simulator with new ordinary three-digit seeds. They must not be selected from the released interventional arrays. This preserves the one-shot test boundary while retaining executable SEM truth.

## Planned CORE use

- T2: leave one complete named SEM family out per fold; select checkpoints on validation families only; report held-out-family performance and retention.
- T4: deterministically render the same generated SCM worlds into `changing_only`, `imagining_only`, and `joint` inputs so the rung-transfer matrix changes one factor at a time.
- CausalT5K adjudication: retain the 9 T2 and 3 T4 accepted task views as supplementary real-text diagnostics, not as the training corpus.

No normalized records have been emitted yet. The next step is a pinned simulator wrapper, deterministic textual renderer, and graph-group-aware manifest.
