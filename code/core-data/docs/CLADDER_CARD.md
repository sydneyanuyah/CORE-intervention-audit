# CLadder data card

Status: eligible deterministic counterfactual conversion complete and verified.

## Source and integrity

- Pinned revision: `not-published`
- Snapshot URL: `https://github.com/causalNLP/cladder/archive/not-published.tar.gz`
- Local raw file: `data/raw/cladder/cladder-3d2d1169.tar.gz`
- Bytes: 9,970,314
- SHA-256: `not-published`
- Balanced questions: 10,112
- Meta-models: 7,064
- Licence: MIT License

The selected `cladder-v1-q-balanced.json` rows link to `cladder-v1-meta-models.json` by `question.meta.model_id == model.model_id`; all links resolve. Each DAG is parsed from the linked model's `structure` string. It is not embedded directly in the question.

## Conversion counts

- Accepted: 1,422 `det-counterfactual` records
- Rejected/quarantined: 8,690 records
- Deterministic oracle mismatches: 0

| Quarantined query type | Rows |
|---|---:|
| `ate` | 1,422 |
| `backadj` | 1,580 |
| `collider_bias` | 158 |
| `correlation` | 1,422 |
| `ett` | 1,264 |
| `exp_away` | 158 |
| `marginal` | 1,580 |
| `nde` | 316 |
| `nie` | 790 |

The quarantined types do not all encode one before/after world: they include observational associations and marginals, adjustment-set selection, population treatment contrasts, nested mediation effects, and collider/conditioning questions. Converting them to one `do(target=value)` would invent or collapse semantics.

## Field mapping and reconstruction

| CORE field | CLadder source or derivation |
|---|---|
| `id` | `cladder-det-counterfactual-<question_id>` |
| `source_id` | balanced row `question_id` |
| `graph` | nodes and edges parsed from linked model `structure` |
| `chain` | `null`; this is a DAG source |
| `factual.passage` | question `given_info`, verbatim |
| `factual.question` | source `question`, verbatim |
| `factual.state` | deterministic SCM solution using non-treatment root evidence and the original treatment value `1-action` |
| `factual.answer` | `null`; source answer concerns the counterfactual question |
| intervention | structural target=`meta.treatment`, exact natural-language `target_text`/`target_span` from `given_info`, value=`meta.action`, canonical `value_token`=`yes`/`no`, kind=`value_set` |
| `intervention.text` | source question, verbatim |
| `intervened.state` | deterministic SCM solution with the same root evidence and treatment overridden to `action` |
| `intervened.answer` | source yes/no answer |
| descendants | directed graph reachability from treatment |
| non-descendants | all other graph nodes except target and descendants |
| probes | one Boolean before/after state probe for every graph node |

The solver evaluates the released deterministic conditional tables in topological order. In 632 rows `X` is endogenous; matching CLadder's generator requires overriding `X` while holding its root causes fixed. For every accepted row, the reconstructed intervened outcome was compared with `meta.polarity`; all 1,422 results matched both `meta.groundtruth` and the released yes/no answer.

## Splits

The pinned release provides no train/validation/test labels. All questions sharing a `model_id` stay together. Assignment is `sha256('cladder-split-v1:' + model_id) mod 100`: buckets 0–79 train, 80–89 validation, and 90–99 test.

- Train: 1,109 records across 428 models
- Validation: 147 records across 60 models
- Test: 166 records across 51 models
- Cross-split model overlap: zero

## Caveats

- Accepted coverage is 1,422 rather than the complete 10,112 balanced questions because the fixed CORE schema represents a single state intervention.
- Graph and state keys use CLadder's symbolic node IDs (`X`, `Y`, `V1`, etc.); natural-language meanings remain in the passage and question.
- Pointer addressing uses the exact natural-language treatment surface in the passage while preserving the symbolic graph node as `intervention.target`.
- The target probe always changes because the released counterfactual contrasts Boolean `action` with `1-action`; downstream nodes may legitimately remain unchanged.
- The yes/no answer asks whether the intervened outcome equals the queried polarity. The actual Boolean outcome is available in `intervened.state`.

## Licence text, verbatim

```text
MIT License

Copyright (c) 2023 CausalNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## First balanced raw item, verbatim fields

```json
{
  "question_id": 4,
  "desc_id": "alarm-mediation-nde-model0-spec0-q0",
  "given_info": "For husbands that don't set the alarm and wives that don't set the alarm, the probability of ringing alarm is 8%. For husbands that don't set the alarm and wives that set the alarm, the probability of ringing alarm is 54%. For husbands that set the alarm and wives that don't set the alarm, the probability of ringing alarm is 41%. For husbands that set the alarm and wives that set the alarm, the probability of ringing alarm is 86%. For husbands that don't set the alarm, the probability of alarm set by wife is 74%. For husbands that set the alarm, the probability of alarm set by wife is 24%.",
  "question": "If we disregard the mediation effect through wife, would husband positively affect alarm clock?",
  "answer": "yes",
  "meta": {
    "story_id": "alarm",
    "graph_id": "mediation",
    "mediators": [
      "V2"
    ],
    "polarity": true,
    "groundtruth": 0.32238166444019534,
    "query_type": "nde",
    "rung": 3,
    "formal_form": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]",
    "given_info": {
      "p(Y | X, V2)": [
        [
          0.08430222457648505,
          0.5394610521458689
        ],
        [
          0.4061509701126924,
          0.8620283206949241
        ]
      ],
      "p(V2 | X)": [
        0.7416866188819116,
        0.23519324071521291
      ]
    },
    "estimand": "\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]",
    "treatment": "X",
    "outcome": "Y",
    "model_id": 0
  },
  "reasoning": {
    "step0": "Let X = husband; V2 = wife; Y = alarm clock.",
    "step1": "X->V2,X->Y,V2->Y",
    "step2": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]",
    "step3": "\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]",
    "step4": "P(Y=1 | X=0, V2=0) = 0.08\nP(Y=1 | X=0, V2=1) = 0.54\nP(Y=1 | X=1, V2=0) = 0.41\nP(Y=1 | X=1, V2=1) = 0.86\nP(V2=1 | X=0) = 0.74\nP(V2=1 | X=1) = 0.24",
    "step5": "0.74 * (0.86 - 0.41) + 0.24 * (0.54 - 0.08) = 0.32",
    "end": "0.32 > 0"
  }
}
```

## Reproducible random spot checks

Five records were sampled from the 1,422 accepted rows with Python `random.Random(20260903)`.

### Spot check 1: `cladder-det-counterfactual-030068`

```json
{
  "chain": null,
  "descendants": [
    "V3",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "We know that cwoi causes yomx. yomx causes gwet. cwoi or gwet causes xevu. We observed an individual is cwoi.",
    "question": "Would an individual is xevu if yomx instead of not yomx?",
    "state": {
      "V1": 1,
      "V3": 0,
      "X": 0,
      "Y": 1
    }
  },
  "graph": {
    "edges": [
      [
        "V1",
        "X"
      ],
      [
        "X",
        "V3"
      ],
      [
        "V1",
        "Y"
      ],
      [
        "V3",
        "Y"
      ]
    ],
    "nodes": [
      "V1",
      "V3",
      "X",
      "Y"
    ]
  },
  "id": "cladder-det-counterfactual-030068",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "V1": 1,
      "V3": 1,
      "X": 1,
      "Y": 1
    }
  },
  "intervention": {
    "formal": "do(X = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      25,
      29
    ],
    "target_text": "yomx",
    "text": "Would an individual is xevu if yomx instead of not yomx?",
    "value": 1,
    "value_token": "yes"
  },
  "non_descendants": [
    "V1"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "must not change",
      "variable": "V1"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "V3"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "X"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "Y"
    }
  ],
  "provenance": {
    "converter": "src/convert_cladder.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
    "sha256": "not-published",
    "url": "https://github.com/causalNLP/cladder/archive/not-published.tar.gz"
  },
  "source": "cladder",
  "source_id": 30068,
  "structure_kind": "dag"
}
```

Assessment: all 1 non-descendant probes are preserved; 2 of 3 target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.

### Spot check 2: `cladder-det-counterfactual-020353`

```json
{
  "chain": null,
  "descendants": [
    "V2",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "We know that smoking causes being lazy. smoking and being hard-working causes being lactose intolerant.",
    "question": "Would the student is not lactose intolerant if smoking instead of nonsmoking?",
    "state": {
      "V2": 1,
      "X": 0,
      "Y": 0
    }
  },
  "graph": {
    "edges": [
      [
        "X",
        "V2"
      ],
      [
        "X",
        "Y"
      ],
      [
        "V2",
        "Y"
      ]
    ],
    "nodes": [
      "V2",
      "X",
      "Y"
    ]
  },
  "id": "cladder-det-counterfactual-020353",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "V2": 0,
      "X": 1,
      "Y": 0
    }
  },
  "intervention": {
    "formal": "do(X = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      13,
      20
    ],
    "target_text": "smoking",
    "text": "Would the student is not lactose intolerant if smoking instead of nonsmoking?",
    "value": 1,
    "value_token": "yes"
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": 0,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "V2"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "X"
    },
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "Y"
    }
  ],
  "provenance": {
    "converter": "src/convert_cladder.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
    "sha256": "not-published",
    "url": "https://github.com/causalNLP/cladder/archive/not-published.tar.gz"
  },
  "source": "cladder",
  "source_id": 20353,
  "structure_kind": "dag"
}
```

Assessment: all 0 non-descendant probes are preserved; 2 of 3 target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.

### Spot check 3: `cladder-det-counterfactual-010284`

```json
{
  "chain": null,
  "descendants": [
    "V2",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "We know that smoking causes absence of tar deposit, and we know that high tar deposit causes lung cancer.",
    "question": "Would the person has lung cancer if nonsmoking instead of smoking?",
    "state": {
      "V2": 0,
      "X": 1,
      "Y": 0
    }
  },
  "graph": {
    "edges": [
      [
        "X",
        "V2"
      ],
      [
        "V2",
        "Y"
      ]
    ],
    "nodes": [
      "V2",
      "X",
      "Y"
    ]
  },
  "id": "cladder-det-counterfactual-010284",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "V2": 1,
      "X": 0,
      "Y": 1
    }
  },
  "intervention": {
    "formal": "do(X = 0)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      13,
      20
    ],
    "target_text": "smoking",
    "text": "Would the person has lung cancer if nonsmoking instead of smoking?",
    "value": 0,
    "value_token": "no"
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "V2"
    },
    {
      "actually_changed": true,
      "answer_after": 0,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "X"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "Y"
    }
  ],
  "provenance": {
    "converter": "src/convert_cladder.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
    "sha256": "not-published",
    "url": "https://github.com/causalNLP/cladder/archive/not-published.tar.gz"
  },
  "source": "cladder",
  "source_id": 10284,
  "structure_kind": "dag"
}
```

Assessment: all 0 non-descendant probes are preserved; 3 of 3 target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.

### Spot check 4: `cladder-det-counterfactual-030812`

```json
{
  "chain": null,
  "descendants": [
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "We know that kraz or hwax causes pexu. kraz or pexu causes rukz. We observed an individual is hwax and an individual is kraz.",
    "question": "Would an individual is rukz if pexu instead of not pexu?",
    "state": {
      "V1": 1,
      "V2": 1,
      "X": 0,
      "Y": 1
    }
  },
  "graph": {
    "edges": [
      [
        "V1",
        "X"
      ],
      [
        "V2",
        "X"
      ],
      [
        "V1",
        "Y"
      ],
      [
        "X",
        "Y"
      ]
    ],
    "nodes": [
      "V1",
      "V2",
      "X",
      "Y"
    ]
  },
  "id": "cladder-det-counterfactual-030812",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "V1": 1,
      "V2": 1,
      "X": 1,
      "Y": 1
    }
  },
  "intervention": {
    "formal": "do(X = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      33,
      37
    ],
    "target_text": "pexu",
    "text": "Would an individual is rukz if pexu instead of not pexu?",
    "value": 1,
    "value_token": "yes"
  },
  "non_descendants": [
    "V1",
    "V2"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "must not change",
      "variable": "V1"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "must not change",
      "variable": "V2"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "X"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "Y"
    }
  ],
  "provenance": {
    "converter": "src/convert_cladder.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
    "sha256": "not-published",
    "url": "https://github.com/causalNLP/cladder/archive/not-published.tar.gz"
  },
  "source": "cladder",
  "source_id": 30812,
  "structure_kind": "dag"
}
```

Assessment: all 2 non-descendant probes are preserved; 1 of 2 target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.

### Spot check 5: `cladder-det-counterfactual-020143`

```json
{
  "chain": null,
  "descendants": [
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "We know that confounder active and assignment of drug treatment causes taking of all assigned drugs. confounder active or taking of all assigned drugs causes freckles. We observed the patient is not assigned the drug treatment and confounder inactive.",
    "question": "Would the patient has freckles if taking of all assigned drugs instead of not taking of any assigned drugs?",
    "state": {
      "V1": 0,
      "V2": 0,
      "X": 0,
      "Y": 0
    }
  },
  "graph": {
    "edges": [
      [
        "V1",
        "X"
      ],
      [
        "V2",
        "X"
      ],
      [
        "V1",
        "Y"
      ],
      [
        "X",
        "Y"
      ]
    ],
    "nodes": [
      "V1",
      "V2",
      "X",
      "Y"
    ]
  },
  "id": "cladder-det-counterfactual-020143",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "V1": 0,
      "V2": 0,
      "X": 1,
      "Y": 1
    }
  },
  "intervention": {
    "formal": "do(X = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      71,
      99
    ],
    "target_text": "taking of all assigned drugs",
    "text": "Would the patient has freckles if taking of all assigned drugs instead of not taking of any assigned drugs?",
    "value": 1,
    "value_token": "yes"
  },
  "non_descendants": [
    "V1",
    "V2"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "must not change",
      "variable": "V1"
    },
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "must not change",
      "variable": "V2"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "X"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "Y"
    }
  ],
  "provenance": {
    "converter": "src/convert_cladder.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
    "sha256": "not-published",
    "url": "https://github.com/causalNLP/cladder/archive/not-published.tar.gz"
  },
  "source": "cladder",
  "source_id": 20143,
  "structure_kind": "dag"
}
```

Assessment: all 2 non-descendant probes are preserved; 2 of 2 target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.
