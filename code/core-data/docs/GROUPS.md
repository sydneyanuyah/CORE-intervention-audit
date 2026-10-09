# Record group sidecars

Scientific resampling and distributed evaluation use true graph/world groups,
not individual records. Group identity is stored outside the immutable record
schema in `data/groups/<source>.jsonl`. Each JSON line contains exactly:

```json
{"graph_group_id": "...", "record_id": "...", "world_group_id": "..."}
```

Every accepted CCR.GB, CLadder, WIQA, and Com² record has exactly one mapping.
The verifier rejects missing, duplicate, unknown, or malformed mappings. It
also rejects any graph or world group whose records occur in more than one of
the train, validation, and test manifests.

Source identities are constructed without changing normalized records:

| Source | Graph group | World group |
|---|---|---|
| CLadder | source `model_id` | `model_id` plus deterministic factual-state digest |
| WIQA | source `graph_id` | `graph_id` plus source question ID |
| CCR.GB | source `context_id` | source `context_id` plus `sample_id` |
| Com² | SHA-256 of the exact first/root chain event | SHA-256 of the complete factual chain |

Com² uses hashes so arbitrary Unicode event text does not become a path-like or
delimiter-ambiguous identifier. Equality of the underlying exact root/chain is
preserved. Sidecars are sorted by `record_id` and rewritten atomically, making
regeneration byte-deterministic.
