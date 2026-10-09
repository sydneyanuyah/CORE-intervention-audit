# Corr2Cause data card

Status: zero CORE intervention records accepted, but an independent validation-only T3 native-classification sidecar is now executable. The two uses are deliberately kept separate.

## Source and licence

- Pinned source: https://huggingface.co/datasets/causal-nlp/corr2cause/tree/not-published
- Official project repository: https://github.com/causalNLP/corr2cause
- Licence: the official code repository contains MIT; the pinned Hugging Face dataset card does not separately declare dataset licence metadata.
- Raw inventory/hashes: paraphrased test `not-published`; clean development `not-published`; paraphrased development `not-published`; refactorized development `not-published`.

## Counts and decision

- Raw/candidate questions: 2,246
- Accepted: 0
- Quarantined: 2,246

Premise/hypothesis/relation classification tests whether correlations identify a causal relation. It has no explicit action and no before/after world pair.

No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.

## T3 native robustness sidecar

`src/prepare_corr2cause_t3.py` preserves Corr2Cause as binary causal-relation classification rather than fabricating interventions. It aligns all 1,076 unique clean development examples to the official reverse-alphabet refactorization and paraphrased development files by exact refactorized text. The official perturbation files contain 2,246 rows; 353 duplicate groups collapse only after checking that released perturbation text and labels are identical.

The generated, uncommitted sidecar contains 1,076 clean/refactorized/paraphrased validation triples across 22 MEC graph groups, with 937 label-0 and 139 label-1 examples. Its manifest SHA-256 is `not-published`; `test_evaluated=false`. This sidecar is valid for T3 robustness only and does not change the zero-row CORE intervention conversion.

## Full raw example

~~~json
{
  "premise": "Suppose there is a closed system of 2 variables, A and B. All the statistical relations among these 2 variables are as follows: A correlates with B.",
  "hypothesis": "A directly affects B.",
  "relation": "neutral",
  "id": "num_nodes=2__mec_id=1__node_i=1__node_j=2__causal_relation=parent__prob=0.50"
}
~~~

## Reproducible quarantine spot checks

There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random(20260903)`; each retains its exact rejection reason so the zero-record decision is auditable.

### Candidate 1

~~~json
{
  "reason": "correlation-to-causal-relation classification has no intervention or paired world states",
  "record": {
    "hypothesis": "A directly affects E.",
    "id": "num_nodes=6__mec_id=1295__node_i=5__node_j=1__causal_relation=child__prob=0.00",
    "premise": "Suppose there is a closed system of 6 variables, A, B, C, D, E and F. All the statistical relations among these 6 variables are as follows: A correlates with C. A correlates with E. A correlates with F. B correlates with C. B correlates with E. B correlates with F. C correlates with E. C correlates with F. D correlates with E. D correlates with F. E correlates with F. However, A is independent of B. A and B are independent given D. A is independent of D. A and D are independent given B. A and D are independent given B and C. A and D are independent given B, C and E. A and D are independent given B, C, E and F. A and D are independent given C. A and D are independent given C and E. A and D are independent given C, E and F. A and E are independent given B and C. A and E are independent given B, C and D. A and E are independent given C. A and E are independent given C and D. B is independent of D. B and D are independent given A. B and D are independent given A and C. B and D are independent given A, C and E. B and D are independent given A, C, E and F. B and D are independent given C. B and D are independent given C and E. B and D are independent given C, E and F. B and E are independent given A and C. B and E are independent given A, C and D. B and E are independent given C. B and E are independent given C and D. C is independent of D. C and D are independent given A. C and D are independent given A and B. C and D are independent given B. C and F are independent given A, B, D and E. C and F are independent given A, B and E. D and F are independent given A, B, C and E. D and F are independent given A, B and E. D and F are independent given A, C and E. D and F are independent given B, C and E. D and F are independent given C and E.",
    "relation": "contradiction"
  }
}
~~~

### Candidate 2

~~~json
{
  "reason": "correlation-to-causal-relation classification has no intervention or paired world states",
  "record": {
    "hypothesis": "E influences A through some mediator(s).",
    "id": "num_nodes=5__mec_id=55__node_i=1__node_j=5__causal_relation=non-child_descendant__prob=0.00",
    "premise": "Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A is independent of B. B and D are independent given A and C. B and D are independent given A, C and E. C and E are independent given A and B. C and E are independent given A, B and D. D and E are independent given A and B. D and E are independent given A, B and C. D and E are independent given A and C.",
    "relation": "contradiction"
  }
}
~~~

### Candidate 3

~~~json
{
  "reason": "correlation-to-causal-relation classification has no intervention or paired world states",
  "record": {
    "hypothesis": "C directly affects B.",
    "id": "num_nodes=5__mec_id=20__node_i=2__node_j=3__causal_relation=child__prob=0.38",
    "premise": "Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with B. A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, B and E are independent given A. B and E are independent given A and C. B and E are independent given A, C and D. B and E are independent given A and D. C and D are independent given A and B. C and D are independent given A, B and E. C and E are independent given A. C and E are independent given A and B. C and E are independent given A, B and D. C and E are independent given A and D. D and E are independent given A. D and E are independent given A and B. D and E are independent given A, B and C. D and E are independent given A and C.",
    "relation": "neutral"
  }
}
~~~

### Candidate 4

~~~json
{
  "reason": "correlation-to-causal-relation classification has no intervention or paired world states",
  "record": {
    "hypothesis": "C influences E through some mediator(s).",
    "id": "num_nodes=5__mec_id=128__node_i=5__node_j=3__causal_relation=non-child_descendant__prob=0.67",
    "premise": "Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with B. A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A and C are independent given B. B and D are independent given A and C. B and D are independent given A, C and E. B and E are independent given A and C. B and E are independent given A, C and D.",
    "relation": "neutral"
  }
}
~~~

### Candidate 5

~~~json
{
  "reason": "correlation-to-causal-relation classification has no intervention or paired world states",
  "record": {
    "hypothesis": "C influences A through some mediator(s).",
    "id": "num_nodes=6__mec_id=760__node_i=1__node_j=3__causal_relation=non-child_descendant__prob=0.50",
    "premise": "Suppose there is a closed system of 6 variables, A, B, C, D, E and F. All the statistical relations among these 6 variables are as follows: A correlates with B. A correlates with C. A correlates with D. A correlates with E. A correlates with F. B correlates with C. B correlates with D. B correlates with E. B correlates with F. C correlates with D. C correlates with E. C correlates with F. D correlates with E. D correlates with F. E correlates with F. However, A and C are independent given B. A and C are independent given B and D. B and D are independent given A. B and D are independent given A and C. B and D are independent given A, C and E. B and D are independent given A, C, E and F. B and D are independent given A, C and F. B and D are independent given A and F. B and E are independent given A and C. B and E are independent given A, C and D. B and E are independent given A, C, D and F. B and E are independent given A, C and F. B and F are independent given A and C. B and F are independent given A, C and D. B and F are independent given A, C, D and E. B and F are independent given A, C and E. C and D are independent given A. C and D are independent given A and B. C and D are independent given A, B and F. C and D are independent given A and F. C and D are independent given B. D and F are independent given A. D and F are independent given A and B. D and F are independent given A, B and C. D and F are independent given A, B, C and E. D and F are independent given A and C. D and F are independent given A, C and E. E and F are independent given A, B and C. E and F are independent given A, B, C and D. E and F are independent given A and C. E and F are independent given A, C and D.",
    "relation": "neutral"
  }
}
~~~
