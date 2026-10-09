# Dataset options for T2, T3, T4, and G4

Date: 2026-09-06

## Decision summary

| Source | Best use | Fit | Annotation burden | Decision |
|---|---|---|---|---|
| CLadder official variants | T3; secondary T4 | Exact natural-language causal benchmark with executable SCMs and easy/hard/common-sense/anti-common-sense/nonsense variants. | None for generated truth. | Use now; canonical raw archive is already present. |
| Corr2Cause development perturbations | T3 | Official clean, reverse-alphabet, and paraphrased development data support native robustness measurement. | None. | Acquired and aligned: 1,076 clean validation examples. |
| CSuite | T2 and T4 | True graphs plus observational and paired interventional environments from known SEMs. It is structured numeric data, not text. | No human labels; requires a deterministic textual renderer and a frozen discretization/query contract. | Best next formal acquisition for executable intervention/counterfactual truth. |
| CausalT5K | T2 domain coverage; T4 diagnostics | Ten domains and L2/L3 questions, but the current release does not provide complete machine-readable SCM states for the CORE contract. | Human or program-assisted structural annotation is required for formal CORE use. | Already acquired; annotate a bounded, double-checked subset rather than inventing fields. |
| Evidence Inference 2.0 | G4 | Real RCT articles with intervention/comparator/outcome prompts, direction labels, and supporting evidence. | No new outcome labels; a protocol amendment is needed because it is trial-level comparative evidence, not a complete DAG world state. | Strongest text-native biomedical addition. |
| CausalBench perturb-seq | G4 | Real genetic interventions and measured single-cell outcomes in K562 and RPE1. | No human annotation; substantial numeric-to-text/task adapter required. | Strongest genuinely interventional biomedical source, but a separate G4 data modality. |
| BioCause | G4 auxiliary training | Biomedical cause/effect spans and triggers in open-access infectious-disease articles. | Existing annotations; not intervention outcomes. | Useful auxiliary extraction data, not formal G4 intervention evidence. |
| CausalWorld / CausalTriplet | T4 research extension | Executable or paired visual/robotic interventions. | No truth annotation, but large modality and model changes. | Do not mix into the current BERT-text family; reserve for a later multimodal extension. |

## Primary-source evidence

- CLadder provides 10,112 balanced questions, executable causal models, deterministic counterfactuals, and official semantic/nonsense variants: https://github.com/causalNLP/cladder
- Corr2Cause explicitly ships `data_2class_from_Z` and paraphrased robustness data and is a causal-relation classification benchmark: https://github.com/causalNLP/corr2cause
- CSuite releases the true adjacency matrix, observational samples, and primary/reference interventional environments: https://github.com/microsoft/csuite
- CausalT5K covers ten domains and Pearl L2/L3 cases under CC-BY-4.0: https://github.com/eyuchang/CausalT5kBench
- Evidence Inference 2.0 provides RCT intervention/comparator/outcome prompts and supporting evidence: https://github.com/jayded/evidence-inference
- CausalBench uses more than 200,000 perturbational single-cell samples from K562 and RPE1: https://github.com/causalbench/causalbench
- BioCause contains 851 annotated biomedical causal relations: https://argo.nactem.ac.uk/biocause/

## Recommended acquisition order

1. Finish the T3 evaluator with the now-prepared CLadder and Corr2Cause development pairs.
2. Pin and convert CSuite into generated T2/T4 text questions with executable ground truth; this is engineering, not annotation.
3. Add Evidence Inference 2.0 as an explicitly amended text-native G4 arm.
4. Add CausalBench only as a separately named perturbational-biology arm; do not pretend expression matrices are ordinary prose examples.
5. Continue the existing CausalT5K annotation protocol for cross-domain coverage.

METER, CounterBench, CRASS, and PubMedCausal remain useful diagnostics, but none alone supplies the complete paired graph-state contract required by the present formal T2/T4/G4 definitions.
