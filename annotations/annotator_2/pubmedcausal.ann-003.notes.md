# PubMedCausal G4 manual review ledger — ann-003

## Scope and method

- Reviewed only `pubmedcausal.ann-003.jsonl` and the train/validation candidates contained in it.
- Did not open or inspect held-out PubMedCausal test data.
- Read `00_START_HERE.md`, the complete annotation Bible, the full annotation guide, and all five PubMedCausal worked examples before annotation.
- Reviewed every candidate sentence and released cause/effect relation individually against the G4 requirement: a source-grounded graph, complete factual state, surgical intervention, and complete comparable intervened state.
- Preserved all immutable identity, source, locator, grouping, and payload fields.

## Row ranges reviewed

- PMC-001–PMC-020: reviewed individually; all reject.
- PMC-021–PMC-040: reviewed individually; all reject.
- PMC-041–PMC-060: reviewed individually; all reject.
- PMC-061–PMC-080: reviewed individually; all reject.
- PMC-081–PMC-100: reviewed individually; all reject.
- PMC-101–PMC-120: reviewed individually; all reject.
- PMC-121–PMC-140: reviewed individually; all reject.
- PMC-141–PMC-160: reviewed individually; all reject.
- PMC-161–PMC-180: reviewed individually; all reject.
- PMC-181–PMC-200: reviewed individually; all reject.
- PMC-201–PMC-220: reviewed individually; all reject.
- PMC-221–PMC-240: reviewed individually; all reject.
- PMC-241–PMC-260: reviewed individually; all reject.
- PMC-261–PMC-280: reviewed individually; all reject.
- PMC-281–PMC-300: reviewed individually; all reject.

## Decision rationale

All 300 released items are sentence-only relation candidates. None contains a complete, source-grounded factual state and complete comparable intervened state. Accordingly, no item satisfies the G4 evidence contract, even where the prose uses causal language, reports a direction or statistical significance, names an experimental manipulation, mentions a comparator, or recommends an intervention. Unsupported `graph`, `factual_state`, `intervention`, and `intervened_state` fields remain null.

The common rejection codes are `FACTUAL_STATE_NOT_IDENTIFIED` and `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`. These codes capture the decisive missing evidence without treating a released cause phrase as an automatically surgical intervention target.

## Difficult and borderline calls

- PMC-021 (hydralazine), PMC-044 (ADP-treated neurons), PMC-068 (Untire access), PMC-073 (streptozotocin), PMC-086/PMC-211 (ST-deficient model), PMC-101 (interoceptive training), PMC-127 (TXA), PMC-134 (CUMS stress model), PMC-151 (parenting intervention), PMC-154 (threat/positive information), PMC-192 (advertisement exposure), PMC-199 (PDC vs controls), PMC-219/PMC-297 (alcohol/cannabis/placebo or aggression exposure), PMC-237 (vitamin C), PMC-244 (question types), PMC-280 (pregabalin), and PMC-291 (menthol packaging) describe manipulations, treatments, or comparators. They still report only qualitative/directional results or incomplete outcome fragments, not complete comparable world states with a grounded surgical target and values.
- PMC-148 includes counterfactual wording (“had the pandemic not occurred”) but does not state a complete alternative world or comparable values.
- PMC-196 reports no significant genetic causal association; this is not an accepted no-effect intervention pair because the sentence lacks a grounded intervention and complete states.
- PMC-255 reports contradictory increasing and decreasing effects. The source is not a complete paired experiment and does not resolve the contradictory state; rejection is safer than manufacturing an adjudication target.
- PMC-297 mentions placebo and multiple treatment conditions, but supplies no complete arm values or endpoint/timepoint state representation.
- Several MR/genetic-causality rows use phrases such as “causal association,” “genetic evidence,” or “causal effect.” Per the guide, these do not identify surgical interventions or paired worlds and were rejected.
- Several cross-sectional/observational rows explicitly warn that causality cannot be inferred. They were rejected for missing paired states rather than converting methodological limitations into graph evidence.

## Completion

- Total reviewed: 300
- Accept: 0
- Reject: 300
- Needs adjudication: 0
- Annotator ID: `ann-003`

## Manual QA pass

- Completed a second full-range QA pass over PMC-001–PMC-300.
- Confirmed exactly 300 rows, continuous boundary IDs PMC-001 through PMC-300, and one completed G4 view per row.
- Confirmed all 300 G4 views use annotator ID `ann-003` and decision `reject`; no null annotator IDs or null decisions remain.
- Confirmed every unsupported structured field remains null: `graph`, `factual_state`, `intervention`, and `intervened_state` each occur as null in all 300 rows.
- Confirmed the edit was confined to `task_views.G4`; immutable `pilot_id`, `source`, `source_locator`, `graph_group_id`, and `source_payload` content was preserved from the assigned queue.
- Reviewed evidence-span handling. The queue supplies released cause/effect labels, but not frozen half-open offsets, and numerous labels contain capitalization changes, inflection changes, misspellings, encoding corruption, or paraphrases relative to the sentence. Because all rows are rejections and the contract forbids guessed offsets, `evidence_spans` remains empty rather than recording false exact spans. The full released sentence and relation candidate were reviewed for every decision.
