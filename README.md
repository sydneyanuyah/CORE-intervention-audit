# CORE

CORE studies when an edit to a language model behaves like a causal intervention: it should make the requested change, preserve unrelated information, and avoid trivial shortcuts.

The repository contains the paper, experiment code, transformed data materials, human annotations, results, and audit analyses. The experiments cover BERT-base, Qwen-7B, Phi-14B, Qwen-32B, and Llama-70B. The registry contains 23 registered analyses/questions.

## Contents

| Path | Contents |
|---|---|
| `paper/` | Current paper. |
| `code/core-models/` | Model training, evaluation, experiment registries, tests, and results. |
| `code/core-data/` | Dataset conversion, generation, validation, schemas, and data cards. |
| `annotations/` | The 900-example annotation sample, two annotation sets, adjudication, and guidelines. |
| `reviewer-audits/` | Additional controls and audit analyses. |
| `post-submission-checks/` | Independent follow-up analyses. |
| `checkpoints/` | Link to the hosted model files. |

The main results workbook is `code/core-models/reports/CORE_RESULTS_TO_DATE.xlsx`. Machine-readable results are stored in the same reports directory.

## Annotation sample

The sample contains 900 source examples: 300 from CausalT5K, 300 from METER, and 300 from PubMedCausal. Two annotators made 1,200 task-view decisions. Observed agreement was 98.58%, multiclass Cohen's kappa was 0.5092, and adjudication retained 12 task views across 9 source examples.

## Models

Model files are hosted at [sanuyah/CORE-intervention-checkpoints](https://huggingface.co/sanuyah/CORE-intervention-checkpoints).

## License

CORE code and project-created materials are available under the MIT License. External datasets and pretrained models retain their original licenses and access terms.
