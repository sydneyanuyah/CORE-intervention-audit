# CORE two-annotator adjudication report

## Outcome

- 900 production source rows and 1,200 task views were compared.
- Decision disagreements requiring review: 17.
- Two additional CausalT5K T4 accept/accept agreements were overridden because intervention-like content was released under L1; the frozen T4 guide forbids silently changing the source rung.
- Final accepted task views: 12 across 9 CausalT5K source rows.
- Final accepted by task: T2 = 9; T4 = 3; G4 = 0.
- METER: 0/300 accepted. PubMedCausal: 0/300 accepted.
- These are accepted annotation candidates, not yet normalized experimental records.

## Agreement

Overall multiclass Cohen's kappa across all 1,200 task views is **0.5092** (observed agreement 98.6%; chance-expected agreement 97.1%).

| Dataset | Views | Decision agreement | Cohen's kappa | Reason-set agreement |
|---|---:|---:|---:|---:|
| causalt5k | 600 | 586/600 (97.7%) | 0.5533 | 87/600 (14.5%) |
| meter | 300 | 299/300 (99.7%) | 0.0000 | 1/300 (0.3%) |
| pubmedcausal | 300 | 298/300 (99.3%) | 0.0000 | 46/300 (15.3%) |

CausalT5K task-specific kappa: T2 = **0.5788**; T4 = **0.5245**.

METER and PubMedCausal each have kappa 0.0000 despite 99%+ raw agreement because annotator B assigned only `reject`, producing no label variance. This prevalence/kappa paradox must be reported alongside raw agreement; it does not mean the annotators disagreed on every item.

## Accepted candidates

- T2: C5K-034, C5K-139, C5K-179, C5K-208, C5K-238, C5K-239, C5K-244, C5K-249, C5K-256.
- T4: C5K-139, C5K-208, C5K-238.
- G4: none.

## Mandatory repair before conversion

`C5K-200` (D10) and `C5K-208` (D6) share the frozen group string `causalt5k:case:0148` despite being unrelated cases. Before normalization or splitting, derive CausalT5K graph groups from release domain plus case ID. Do not silently edit the adjudicated frozen input.

## Audit cautions

- Both CausalT5K submissions identify themselves as `ann-001`; provenance therefore distinguishes package A and package B by path and SHA-256.
- The first package's pre-existing self-adjudicated files were not used as an independent vote.
- PubMedCausal held-out test data were not accessed.
- Run the packaged submission validator against each adjudicated queue before downstream conversion.
