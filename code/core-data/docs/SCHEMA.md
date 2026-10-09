# CORE normalized record schema

## Contract

Every accepted JSONL line contains exactly the fields defined in `src/schema.py`; unexpected keys are rejected. Unknown source information is represented as JSON `null`, never inferred.

`structure_kind` controls the structural representation:

- `dag`: `graph` contains unique node names and directed edges, the graph must be acyclic, and `chain` is `null`.
- DAG records may additionally include `graph.node_text`, an exact mapping from
  every stable node ID to one or more released natural-language groundings.
  These descriptions are input metadata, not inferred labels.
- `chain`: `chain` contains the source event path and `graph` is `null`. Chain position supplies weaker ordering information, not a full causal graph.

The intervention target is excluded from both `descendants` and `non_descendants`. The two lists must be unique and disjoint.

## Intervention addressing

Every intervention has exactly these fields: `target`, `target_text`,
`target_span`, `value`, `value_token`, `replacement_span`, `kind`, `formal`,
and `text`. Character spans are two-element `[start, end]` arrays using
zero-based, half-open Python slice offsets.

- `target` is the stable structural identifier used by the graph or chain.
  `target_text` is the target's exact surface form in the model input. They may
  differ; converters must not replace a structural ID with display text.
- `target_span` is required for both intervention kinds. It must be within
  `factual.passage`, and `factual.passage[start:end]` must equal `target_text`
  exactly, including case and whitespace.
- For `kind: "value_set"`, `value_token` is a required non-null JSON scalar
  giving the canonical closed-vocabulary value, and `replacement_span` must be
  `null`.
- For `kind: "event_replace"`, `value_token` must be `null` and
  `replacement_span` is required. It must be within `factual.passage`, and
  `factual.passage[start:end]` must equal the string in `value` exactly. The
  factual/model-input passage therefore contains both the old and replacement
  event mentions needed by the pointer.
- Invalid, reversed, empty, non-integer, out-of-bounds, or text-mismatched
  spans are validation errors, not warnings.

These fields support text conditioning and pointer addressing without asking a
model or converter to infer where the intervention appears.

## Probe semantics

A probe is labelled either `must not change` or `may change`.

- Every `must not change` probe must have equal before/after answers and `actually_changed: false`.
- An accepted record must have at least one `may change` probe with a genuine before/after difference.
- A structurally valid record with no observed may-change effect can be detected using `validate_record(..., require_observed_effect=False)` and written to `<source>.rejected.jsonl` as `{ "reason": "...", "record": {...} }`.

## Values and missingness

Answers, intervention values, and state values may be JSON strings, numbers, booleans, or `null`. `NaN` and infinity are forbidden. State is either a variable-to-scalar object or `null`.

## Provenance

Every record stores the fetch time, exact source URL, local raw path, lowercase SHA-256, converter path, and converter version. Fixture hashes are synthetic placeholders used only by unit tests; generated benchmark records must carry the actual raw-file hash.

## Train, validation, and test data

Split membership is stored outside the immutable record contract in `data/splits/<source>.<split>.txt`, one record ID per line. Authoritative source splits are preserved. Any constructed split must be deterministic and group-aware so records sharing a graph, scenario, paragraph, or world cannot leak across splits.

When any split manifests exist, `src/verify.py` requires every accepted record ID to appear in exactly one manifest and rejects unknown or multiply assigned IDs.
