# Data Availability and Licensing Map

Original CORE software and project-created materials are released under the MIT License in `LICENSE`. Upstream datasets and pretrained models retain their own licenses and access terms.

The repository publishes transformation code, schemas, data cards, and the project-created annotation layer. Upstream corpora are included only where redistribution is permitted.

| Source | Role | Public-package treatment |
|---|---|---|
| XOR synthetic | Controlled SCM and route-law tests | Regenerate from project code/manifests. |
| CCR.GB | Graph-structured causal benchmark | GPL-3.0 code is acquired from its authoritative repository; do not vendor incompatible upstream code into a differently licensed release. This release includes only the 20 generated feasibility worlds and their 120 normalized records required by the bundled two-edit unit test. |
| CLadder | Causal question benchmark | Source is MIT-licensed; use the recorded converter and preserve attribution. |
| WIQA | Procedural causal reasoning | Source is Apache-2.0; use the recorded converter and preserve attribution. |
| Com2 | Commonsense causal editing | Consumed from the authors' release. The CORE repository publishes adapters/metadata, not the upstream repository contents unless its license explicitly permits redistribution. |
| CounterBench | External causal evaluation | Fetch from the authoritative release and preserve its license/citation. |
| CRASS | External causal evaluation | Apache-2.0; preserve attribution and the registered subset identity. |
| Evidence Inference 2.0 | Biomedical evidence transfer | Fetch under the upstream terms. Publish only permitted identifiers, transformations, and annotations. |
| CausalT5K | Annotation and held-out-domain candidate source | Publish the project-created labels and source locators only to the extent permitted by the upstream terms. |
| METER | Rung-transfer annotation candidate source | Publish project-created decisions and locators; respect upstream text redistribution terms. |
| PubMedCausal | Biomedical causal-relation annotation candidate source | The annotation queue contains train/validation candidates only; held-out test data were not accessed. Respect upstream text and PubMed terms. |

Source-specific acquisition, schema, and conversion notes are in `code/core-data/docs/`. The annotation sample and both annotation outputs are under `annotations/`.

## Annotation release statistics

- Selection seed: 317.
- Production source rows: 900 (300 per dataset).
- Worked examples: 15 (5 per dataset), excluded from production.
- Task-view decisions: 1,200.
- Overall observed decision agreement: 98.58%.
- Overall multiclass Cohen's kappa: 0.5092.
- Final accepted views: 12 across 9 CausalT5K source rows.
- PubMedCausal queue: 274 train and 26 validation candidates; no held-out test access.

For METER and PubMedCausal, zero kappa accompanies greater-than-99% raw agreement because one annotator used only the reject label, producing the prevalence/kappa paradox.
