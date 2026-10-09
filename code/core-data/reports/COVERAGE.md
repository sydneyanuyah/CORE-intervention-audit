# CORE record coverage

All four priority sources are converted to the extent supported by the fixed record contract. All six lower-priority sources were also pinned, downloaded, inspected, and passed through source-specific converters. They contribute no accepted records because none exposes enough paired intervention evidence for this contract. Synthetic unit-test fixtures are excluded from benchmark coverage.

| Source | Records | Structure kind | Mean probes | Mean non-descendants | Licence | Caveat |
|---|---:|---|---:|---:|---|---|
| Com2 | 770 | chain | 5.0117 | 0.4182 | none stated | 448/500 counterfactual rows have no preservation probe; path structure is weaker than a DAG |
| CLadder | 1,422 | DAG | 3.5556 | 0.8889 | MIT | 8,690 non-single-intervention queries are explicitly quarantined |
| WIQA | 26,045 | signed DAG | 3.7357 | 2.7357 | Apache-2.0 | direction-of-change only; 13,660 no-effect/unresolved/ambiguous rows quarantined |
| CCR.GB | 36,000 | DAG | 4.0000 | 0.6667 | none stated at pinned commit | 6,000 generated worlds; 36,000 factual-equivalent actions quarantined |

## Assessed sources with zero compatible records

| Source | Raw candidates | Accepted | Quarantined | Licence | Contract blocker |
|---|---:|---:|---:|---|---|
| CounterBench | 1,200 | 0 | 1,200 | MIT metadata | no factual outcome/state pair; 500 rows also have multiple actions |
| CausalT5K | 7,260 | 0 | 7,260 | dataset CC-BY-4.0; code MIT | no machine-readable edge list or paired states; L1 has no intervention |
| Corr2Cause paraphrased test | 2,246 | 0 | 2,246 | none stated at pinned revision | relation classification, not an intervention world pair |
| CRASS | 274 | 0 | 274 | Apache-2.0 | counterfactual multiple choice without graph or paired states |
| METER | 12,445 | 0 | 12,445 | none stated | natural-language questions without graph or paired states |
| PubMedCausal | 33,628 long candidates | 0 | 33,628 | MIT | causal span extraction, not an intervention world pair |

## Verifier output

```text
VERIFY PASS
accepted_files=11
records=64237
rejected_records=115403
unique_ids=64237
split_entries=64237
split_coverage=complete
graph_groups=9311
world_groups=33914
priority_group_coverage=complete
mean_probes_per_record=3.8951
mean_non_descendants_per_record=1.5075
round_trip=pass
```

This output covers the four accepted priority sources plus empty accepted outputs and reason-wrapped quarantines for all six lower-priority sources.
