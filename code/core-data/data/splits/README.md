# Split manifests

Benchmark splits are represented as newline-delimited record IDs in source-specific files:

- `<source>.train.txt`
- `<source>.validation.txt`
- `<source>.test.txt`

Converters must preserve an authoritative source split when one exists. If a source has no split, a deterministic group-aware split rule must be recorded in `logs/DECISIONS.md` and the source data card before manifests are generated. Related records from the same causal graph, scenario, paragraph, or world must not cross splits.

