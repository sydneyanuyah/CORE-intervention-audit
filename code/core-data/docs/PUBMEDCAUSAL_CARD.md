# PubMedCausal data card

Status: raw source acquired and fully assessed; zero CORE records accepted because the release cannot satisfy the fixed paired-intervention contract without invented data.

## Source and licence

- Pinned source: https://huggingface.co/datasets/jaypee01/PubMedCausal/tree/not-published
- Licence: MIT License; `Copyright (c) 2026 edyahlimited` (verbatim opening lines)
- Raw inventory/hashes: `{"30k_test.json": "not-published", "30k_train.json": "not-published", "validation_set.json": "not-published"}`

## Counts and decision

- Raw/candidate questions: 33,628
- Accepted: 0
- Quarantined: 33,628

The 16 wide Cause/Effect groups are reshaped into one long candidate per populated relation. Rows with no relation remain one negative candidate. These are span relations, not interventions or paired world states.

No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.

## Full raw example

~~~json
{
  "s/n": 6350,
  "Sentence": "This study aims to examine the causal relationship between internet usage and BMI among the elderly, addressing a gap in existing research and providing evidence for the development of health policies targeted at the elderly population",
  "Cause 1": "",
  "Effect 1": "",
  "Sententiality 1": "",
  "Causality 1": "",
  "Cause 2": "",
  "Effect 2": "",
  "Sententiality 2": "",
  "Causality 2": "",
  "Cause 3": "",
  "Effect 3": "",
  "Sententiality 3": "",
  "Causality 3": "",
  "Cause 4": "",
  "Effect 4": "",
  "Sententiality 4": "",
  "Causality 4": "",
  "Cause 5": "",
  "Effect 5": "",
  "Sententiality 5": "",
  "Causality 5": "",
  "Cause 6": "",
  "Effect 6": "",
  "Sententiality 6": "",
  "Causality 6": "",
  "Cause 7": "",
  "Effect 7": "",
  "Sententiality 7": "",
  "Causality 7": "",
  "Cause 8": "",
  "Effect 8": "",
  "Sententiality 8": "",
  "Causality 8": "",
  "Cause 9": "",
  "Effect 9": "",
  "Sententiality 9": "",
  "Causality 9": "",
  "Cause 10": "",
  "Effect 10": "",
  "Sententiality 10": "",
  "Causality 10": "",
  "Cause 11": "",
  "Effect 11": "",
  "Sententiality 11": "",
  "Causality 11": "",
  "Cause 12": "",
  "Effect 12": "",
  "Sententiality 12": "",
  "Causality 12": "",
  "Cause 13": "",
  "Effect 13": "",
  "Sententiality 13": "",
  "Causality 13": "",
  "Cause 14": "",
  "Effect 14": "",
  "Sententiality 14": "",
  "Causality 14": "",
  "Cause 15": "",
  "Effect 15": "",
  "Sententiality 15": "",
  "Causality 15": "",
  "Cause 16": "",
  "Effect 16": "",
  "Sententiality 16": "",
  "Causality 16": ""
}
~~~

## Reproducible quarantine spot checks

There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random(20260903)`; each retains its exact rejection reason so the zero-record decision is auditable.

### Candidate 1

~~~json
{
  "reason": "non-causal sentence has no intervention candidate",
  "record": {
    "causality": null,
    "cause": null,
    "effect": null,
    "is_causal": null,
    "relation_index": null,
    "release_split": "test",
    "sentence": "Adiponectin levels increase over time in long-lived adults and are associated with greater physical disability and mortality. Such increases may occur in response to age-related homeostatic dysregulation. Additional investigation is required to define the underlying mechanisms and whether this represents a marker or causal factor for mortality in this age group",
    "sententiality": null,
    "source_id": 1731
  }
}
~~~

### Candidate 2

~~~json
{
  "reason": "non-causal sentence has no intervention candidate",
  "record": {
    "causality": null,
    "cause": null,
    "effect": null,
    "is_causal": null,
    "relation_index": null,
    "release_split": "test",
    "sentence": "Estimate the effectiveness of brief interventions in reducing trauma recidivism in hospitalized trauma patients who screened positive for alcohol and/or illicit drug use",
    "sententiality": null,
    "source_id": 10509
  }
}
~~~

### Candidate 3

~~~json
{
  "reason": "cause/effect span relation has no intervention or paired world states",
  "record": {
    "causality": "Implicit",
    "cause": "postwar stressors",
    "effect": "contribution to the intensity of posttraumatic symptoms",
    "is_causal": null,
    "relation_index": 3,
    "release_split": "train",
    "sentence": "Long-term exposure to war and postwar stressors caused serious psychological consequences in civilian women, with PTSD being only one of the disorders in the wide spectrum of posttraumatic reactions. Postwar stressors did not influence the prevalence of PTSD but they did contribute to the intensity and number of posttraumatic symptoms",
    "sententiality": "Intra",
    "source_id": 28211
  }
}
~~~

### Candidate 4

~~~json
{
  "reason": "non-causal sentence has no intervention candidate",
  "record": {
    "causality": null,
    "cause": null,
    "effect": null,
    "is_causal": null,
    "relation_index": null,
    "release_split": "train",
    "sentence": "Our meta-analysis demonstrated a statistically significant negative association between adequate levels of HL and the likelihood of depression, especially among adolescents. More longitudinal studies with rigorous design are needed to further explore the causal relationship and long-term associations",
    "sententiality": null,
    "source_id": 4066
  }
}
~~~

### Candidate 5

~~~json
{
  "reason": "cause/effect span relation has no intervention or paired world states",
  "record": {
    "causality": "Explicit",
    "cause": "high insulin level",
    "effect": "An increase risk of endometrial cancer",
    "is_causal": null,
    "relation_index": 1,
    "release_split": "test",
    "sentence": "This study provides evidence to support a causal association of higher insulin levels, independently of BMI, with endometrial cancer risk",
    "sententiality": "Intra",
    "source_id": 8084
  }
}
~~~
