# CORE operator integration — 2026-09-04

## Canonical server implementations

- Core O1/O2/O3, laws, placebo, training and evaluation: `${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py`
- Joint-text addressing and held-out wording variants: `${PRIVATE_STORAGE_ROOT}/CORE_joint_text_20graph/core_components.py`
- Graph-masked O3 variants: `${PRIVATE_STORAGE_ROOT}/CORE_o3g/core_components.py`
- Final constrained law selection: `${PRIVATE_STORAGE_ROOT}/CORE_final_law_gate/core_components.py`

The launcher runs independent graph seeds concurrently, matching the graph-level statistical unit: four GPUs for BERT-base and eight GPUs for BERT-large.

## Validated

| Model | Concurrent GPUs | Runs passed | Hidden size | O2 parameters | O3 parameters | Forward/decode/backward/laws |
|---|---:|---:|---:|---:|---:|---|
| BERT-base | 4 | 4/4 | 768 | 255,744 | 1,264,128 | pass |
| BERT-large | 8 | 8/8 | 1,024 | 340,992 | 2,209,792 | pass |

All runs produced `[2,16,H]` edited states and `[2,30,2]` reader logits, finite gradients, and identity/idempotence/commutation/last-write loss values. Peak memory for this minimal integration batch was about 486 MiB for base and 1,397 MiB for large; production batch sizing must be measured on complete graph jobs.

The operator runtime is pinned to the server's `${PRIVATE_STORAGE_ROOT}/DLclass/bin/python` environment (PyTorch 2.11, Transformers 4.57). Transformers 5.16 changed BERT's internal layer contract and is incompatible with the historical split-reader implementation.

## Registry corrections and remaining implementation

- The server has do-nothing and random-init floors. This repository now adds the missing deterministic do-everything prediction floor: every variable receives the requested intervention value, while no-op preserves the factual state.
- O3-G is a graph-slot address supplied by an integer intervention ID. It is not the registry's T3 textual pointer.
- T0 learned ID and T1 joint-text/unseen-paraphrase arms exist.
- T2 inline in-context editing, T3 textual pointer, and correct/adjacent/random span controls still require new modules before A1–A3 can be called complete.
- Historical graph tests have already been accessed. New evidence must use fresh preregistered graph seeds; `dev_graph_3999` is appropriate for development only.

An attempted historical end-to-end foundation smoke was stopped after five minutes because it repeatedly evaluates full train/validation/test sets for every condition and is not appropriate for a fast integration test. No scientific result was taken from it.
