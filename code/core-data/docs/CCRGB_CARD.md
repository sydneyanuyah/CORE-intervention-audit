# CCR.GB data card

Status: 6,000-world external-generator conversion complete and verified.

## Source and isolation

- Repository: https://github.com/jmaasch/compositional_causal_reasoning
- Pinned generator commit: not-published
- External clone: sibling ${USER_HOME}/Documents/compositional_causal_reasoning
- Raw generated output: ${USER_HOME}/Documents/compositional_causal_reasoning/output/ccrgb_6000_worlds.jsonl
- Raw bytes: 48,267,204
- Raw SHA-256: not-published
- Domain: ClinicalNotes
- Generator seed: 20260903
- Licence: none stated at the pinned commit. No LICENSE, COPYING, licence text, or README licence declaration is present.

The generator was cloned and run outside core-data. No CCR.GB source file is copied, imported, or vendored by the shipped converter. Only the generator JSONL output is consumed.

## Generation

The CPU-only run used graph_sizes=[[2,2,2]], cycle BCCs, random monotone Boolean functions, 6,000 tasks/worlds, and one sample per world. It produced 6,000 unique four-node worlds, 36,000 cause-effect pairs, and both Boolean actions for every pair.

## Conversion counts

- Candidate interventions: 72,000
- Accepted: 36,000
- Quarantined: 36,000
- Rejection reason: intervention matches the factual state and therefore has no observed may-change probe

Exactly one of the two Boolean actions per pair differs from the factual cause value. The other reproduces the factual world and is quarantined under the shared observed-effect requirement.

## Field mapping

| CORE field | CCR.GB output source or derivation |
|---|---|
| graph | dag_nodes plus nonzero entries of dag_adjacency_matrix |
| factual.passage | generated causal context joined with sample context |
| factual.state | generated factual query true_endogenous |
| factual.question/answer | matching factual effect prompt and Boolean response mapped to yes/no |
| intervention | counterfactual cause and action 0/1; `target_text` is the exact variable mention in the factual passage, `target_span` addresses it, and `value_token` is canonical `0`/`1` |
| intervened.state | generated action-specific true_endogenous |
| intervened.answer | generated action-specific response mapped to yes/no |
| descendants | DAG reachability from the cause |
| non-descendants | other DAG nodes excluding cause and descendants |
| probes | every endogenous node generated factual/intervened Boolean value |

## Splits

Whole context worlds are kept together. Context IDs are ranked by SHA-256 of `ccrgb-split-v1:<context_id>` and assigned deterministically in an 80/10/10 train/validation/test partition.

- Train: 28800 records
- Validation: 3600 records
- Test: 3600 records
- Train worlds: 4800
- Validation worlds: 600
- Test worlds: 600
- Cross-split world overlap: zero

## Caveats

- All worlds use one domain and the smallest configured graph size; the scale supports repeated graph-level inference but does not establish broad domain coverage.
- Variable names and clinical prose are randomly generated and should not be interpreted as real medical advice or patient data.
- The pinned repository missing licence statement conflicts with the task GPL-3.0 expectation. Redistribution rights must not be inferred.

## First generated raw world, verbatim fields

~~~json
{
  "bernoulli_parameters": [
    0.7,
    0.8,
    0.5,
    0.6
  ],
  "causal_context": "Chronic disease YVR4BL sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 9/10) if they carry allele SEH8, a genetic marker for severe YVR4BL. If the patient self-reports significant pain or the patient has a family history of XAJ8, then lab 2J92 will be elevated (greater than 2.1 mg/dL). If 2J92 is elevated or the patient carries allele J70J, then vital RIMS will be low (less than 1.82 mg/dL). If RIMS is low and the patient has previously received surgery for 12IM, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here.",
  "compositions": [
    [
      [
        "pain",
        "2J92"
      ],
      [
        "2J92",
        "RIMS"
      ],
      [
        "RIMS",
        "surgery"
      ]
    ],
    [
      [
        "pain",
        "2J92"
      ],
      [
        "2J92",
        "surgery"
      ]
    ],
    [
      [
        "pain",
        "RIMS"
      ],
      [
        "RIMS",
        "surgery"
      ]
    ]
  ],
  "context_id": 0,
  "counterfactual_queries": [
    {
      "cause": "pain",
      "cause_false": {
        "prompt": "Now suppose that the patient will not be in pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 0,
          "RIMS": 0,
          "pain": 0,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "surgery"
    },
    {
      "cause": "pain",
      "cause_false": {
        "prompt": "Now suppose that the patient will not be in pain regardless of all other circumstances. With this new assumption, will lab 2J92 be elevated? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 0,
          "RIMS": 0,
          "pain": 0,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will lab 2J92 be elevated? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "2J92"
    },
    {
      "cause": "pain",
      "cause_false": {
        "prompt": "Now suppose that the patient will not be in pain regardless of all other circumstances. With this new assumption, will vital RIMS be low? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 0,
          "RIMS": 0,
          "pain": 0,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will vital RIMS be low? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "RIMS"
    },
    {
      "cause": "2J92",
      "cause_false": {
        "prompt": "Now suppose that lab 2J92 will not be elevated regardless of all other circumstances. With this new assumption, will vital RIMS be low? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 0,
          "RIMS": 0,
          "pain": 1,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that lab 2J92 will be elevated regardless of all other circumstances. With this new assumption, will vital RIMS be low? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "RIMS"
    },
    {
      "cause": "2J92",
      "cause_false": {
        "prompt": "Now suppose that lab 2J92 will not be elevated regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 0,
          "RIMS": 0,
          "pain": 1,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that lab 2J92 will be elevated regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "surgery"
    },
    {
      "cause": "RIMS",
      "cause_false": {
        "prompt": "Now suppose that vital RIMS will not be low regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 0,
          "pain": 1,
          "surgery": 0
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 0
      },
      "cause_true": {
        "prompt": "Now suppose that vital RIMS will be low regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
        "true_endogenous": {
          "2J92": 1,
          "RIMS": 1,
          "pain": 1,
          "surgery": 1
        },
        "true_exogenous": {
          "12IM": 1,
          "J70J": 0,
          "SEH8": 1,
          "XAJ8": 0
        },
        "true_response": 1
      },
      "effect": "surgery"
    }
  ],
  "dag_adjacency_matrix": [
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0
    ]
  ],
  "dag_nodes": [
    "pain",
    "2J92",
    "RIMS",
    "surgery"
  ],
  "exogenous_variables": [
    "SEH8",
    "XAJ8",
    "J70J",
    "12IM"
  ],
  "factual_queries": [
    {
      "effect": "surgery",
      "prompt": "Given these history and physical notes, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
      "true_endogenous": {
        "2J92": 1,
        "RIMS": 1,
        "pain": 1,
        "surgery": 1
      },
      "true_exogenous": {
        "12IM": 1,
        "J70J": 0,
        "SEH8": 1,
        "XAJ8": 0
      },
      "true_response": 1
    },
    {
      "effect": "2J92",
      "prompt": "Given these history and physical notes, will lab 2J92 be elevated? Begin your response with Yes or No and be as concise as possible.",
      "true_endogenous": {
        "2J92": 1,
        "RIMS": 1,
        "pain": 1,
        "surgery": 1
      },
      "true_exogenous": {
        "12IM": 1,
        "J70J": 0,
        "SEH8": 1,
        "XAJ8": 0
      },
      "true_response": 1
    },
    {
      "effect": "RIMS",
      "prompt": "Given these history and physical notes, will vital RIMS be low? Begin your response with Yes or No and be as concise as possible.",
      "true_endogenous": {
        "2J92": 1,
        "RIMS": 1,
        "pain": 1,
        "surgery": 1
      },
      "true_exogenous": {
        "12IM": 1,
        "J70J": 0,
        "SEH8": 1,
        "XAJ8": 0
      },
      "true_response": 1
    }
  ],
  "generator_commit": "not-published",
  "generator_domain": "ClinicalNotes",
  "generator_seed": 20260903,
  "global_quantity": [
    "pain",
    "surgery"
  ],
  "local_quantities": [
    [
      "pain",
      "2J92"
    ],
    [
      "pain",
      "RIMS"
    ],
    [
      "2J92",
      "RIMS"
    ],
    [
      "2J92",
      "surgery"
    ],
    [
      "RIMS",
      "surgery"
    ]
  ],
  "nodes_per_bcc": [
    2,
    2,
    2
  ],
  "sample_context": "Now, we will review the history and physical notes for patient Toni Wright. History of Present Illness: Toni Wright is a 59-year-old female with YVR4BL who presented to the emergency department with acute onset pain that began 5 hours prior to arrival. Pain was rated 9/10. The patient reports the pain has been persistent since onset. The patient took aspirin (100 mg) at home with minimal relief. Genetic Screening: Patient carries alleles SEH8, RGII, AGYE. Family History: 1P19, QGV0. Medications: ML8 50 mg/day. Past Surgical History: Prior surgeries for 12IM, M7VC, VUE4.",
  "sample_id": 0
}
~~~

## Reproducible random spot checks

Five accepted records were sampled with Python random.Random(20260903).

### Spot check 1: ccrgb-clinical-5088-000-03-do0

~~~json
{
  "chain": null,
  "descendants": [
    "ZW4P",
    "surgery"
  ],
  "factual": {
    "answer": "yes",
    "passage": "Chronic disease L5HQNN sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 9/10) if they carry allele 74WG, a genetic marker for severe L5HQNN. If the patient self-reports significant pain or the patient carries allele AAQT, then vital BWGI will be low (less than 1.2 mg/dL). If BWGI is low or the patient carries allele TH5L, then vital ZW4P will be low (less than 2.15 mg/dL). If ZW4P is low and the patient carries allele XDLZ, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here. Now, we will review the history and physical notes for patient Jennifer Gomez. History of Present Illness: Jennifer Gomez is a 57-year-old female with L5HQNN who presented to the emergency department with acute onset pain that began 2 hours prior to arrival. Pain was rated 9/10. The patient reports the pain has been persistent since onset. The patient took aspirin (500 mg) at home with minimal relief. Genetic Screening: Patient carries alleles 74WG, AAQT, TH5L, W8F8, 1F8D. Family History: WXVX, 1QT2. Medications: 22S 25 mg/day. Past Surgical History: Prior surgeries for JR8I, GDU5.",
    "question": "Given these history and physical notes, will vital ZW4P be low? Begin your response with Yes or No and be as concise as possible.",
    "state": {
      "BWGI": 1,
      "ZW4P": 1,
      "pain": 1,
      "surgery": 0
    }
  },
  "graph": {
    "edges": [
      [
        "pain",
        "BWGI"
      ],
      [
        "BWGI",
        "ZW4P"
      ],
      [
        "ZW4P",
        "surgery"
      ]
    ],
    "nodes": [
      "pain",
      "BWGI",
      "ZW4P",
      "surgery"
    ]
  },
  "id": "ccrgb-clinical-5088-000-03-do0",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "BWGI": 0,
      "ZW4P": 1,
      "pain": 1,
      "surgery": 0
    }
  },
  "intervention": {
    "formal": "do(BWGI = 0)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "BWGI",
    "target_span": [
      373,
      377
    ],
    "target_text": "BWGI",
    "text": "Now suppose that vital BWGI will not be low regardless of all other circumstances. With this new assumption, will vital ZW4P be low? Begin your response with Yes or No and be as concise as possible.",
    "value": 0,
    "value_token": "0"
  },
  "non_descendants": [
    "pain"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "must not change",
      "variable": "pain"
    },
    {
      "actually_changed": true,
      "answer_after": 0,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "BWGI"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": "Now suppose that vital BWGI will not be low regardless of all other circumstances. With this new assumption, will vital ZW4P be low? Begin your response with Yes or No and be as concise as possible.",
      "required": "may change",
      "variable": "ZW4P"
    },
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "surgery"
    }
  ],
  "provenance": {
    "converter": "src/convert_ccrgb.py",
    "converter_version": 3,
    "fetched_utc": "2026-09-08T00:00:00Z",
    "file": "ccrgb_6000_worlds.jsonl",
    "sha256": "not-published",
    "url": "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
  },
  "source": "ccrgb",
  "source_id": "5088:0:3:0",
  "structure_kind": "dag"
}
~~~

Assessment: 1 of 4 generated state probes changed and all 1 graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.

### Spot check 2: ccrgb-clinical-2952-000-00-do0

~~~json
{
  "chain": null,
  "descendants": [
    "3MY0",
    "FXUM",
    "surgery"
  ],
  "factual": {
    "answer": "yes",
    "passage": "Chronic disease Q51X6E sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 7/10) if they carry allele LL19, a genetic marker for severe Q51X6E. If the patient self-reports significant pain or the patient has a family history of QRNC, then lab 3MY0 will be elevated (greater than 0.51 mg/dL). If 3MY0 is elevated or the patient has a family history of U6EH, then lab FXUM will be low (less than 2.38 mg/dL). If FXUM is low and the patient has a family history of BZMT, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here. Now, we will review the history and physical notes for patient Jerry Robbins. History of Present Illness: Jerry Robbins is a 63-year-old male with Q51X6E who presented to the emergency department with acute onset pain that began 3 hours prior to arrival. Pain was rated 8/10. The patient reports the pain has been persistent since onset. The patient took aspirin (250 mg) at home with minimal relief. Genetic Screening: Patient carries alleles LL19, U8NK, 30MQ. Family History: U6EH, BZMT, 6QBF, V8EZ. Medications: HX9 25 mg/day. Past Surgical History: Prior surgeries for 3AOA, VWAS.",
    "question": "Given these history and physical notes, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "state": {
      "3MY0": 1,
      "FXUM": 1,
      "pain": 1,
      "surgery": 1
    }
  },
  "graph": {
    "edges": [
      [
        "pain",
        "3MY0"
      ],
      [
        "3MY0",
        "FXUM"
      ],
      [
        "FXUM",
        "surgery"
      ]
    ],
    "nodes": [
      "pain",
      "3MY0",
      "FXUM",
      "surgery"
    ]
  },
  "id": "ccrgb-clinical-2952-000-00-do0",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "3MY0": 0,
      "FXUM": 1,
      "pain": 0,
      "surgery": 1
    }
  },
  "intervention": {
    "formal": "do(pain = 0)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "pain",
    "target_span": [
      175,
      179
    ],
    "target_text": "pain",
    "text": "Now suppose that the patient will not be in pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "value": 0,
    "value_token": "0"
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": 0,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "pain"
    },
    {
      "actually_changed": true,
      "answer_after": 0,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "3MY0"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "FXUM"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": "Now suppose that the patient will not be in pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
      "required": "may change",
      "variable": "surgery"
    }
  ],
  "provenance": {
    "converter": "src/convert_ccrgb.py",
    "converter_version": 3,
    "fetched_utc": "2026-09-08T00:00:00Z",
    "file": "ccrgb_6000_worlds.jsonl",
    "sha256": "not-published",
    "url": "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
  },
  "source": "ccrgb",
  "source_id": "2952:0:0:0",
  "structure_kind": "dag"
}
~~~

Assessment: 2 of 4 generated state probes changed and all 0 graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.

### Spot check 3: ccrgb-clinical-1702-000-05-do1

~~~json
{
  "chain": null,
  "descendants": [
    "surgery"
  ],
  "factual": {
    "answer": "no",
    "passage": "Chronic disease P0XV4A sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 7/10) if they carry allele LCI0, a genetic marker for severe P0XV4A. If the patient self-reports significant pain or the patient carries allele V3HS, then lab LDX4 will be elevated (greater than 2.17 mg/dL). If LDX4 is elevated or the patient has previously received surgery for XMEJ, then lab SZYN will be elevated (greater than 0.42 mg/dL). If SZYN is elevated and the patient has a family history of 107S, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here. Now, we will review the history and physical notes for patient Jeremy Ray. History of Present Illness: Jeremy Ray is a 61-year-old male with P0XV4A who presented to the emergency department with acute onset pain that began 3 hours prior to arrival. Pain was rated 4/10. The patient reports the pain has been persistent since onset. The patient took aspirin (250 mg) at home with minimal relief. Genetic Screening: Patient carries alleles ZDOI, S1CU. Family History: 107S, 73E7, VXWP. Medications: D2P 150 mg/day, GNY 50 mg/day. Past Surgical History: Prior surgeries for OMEY, VBVG.",
    "question": "Given these history and physical notes, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "state": {
      "LDX4": 0,
      "SZYN": 0,
      "pain": 0,
      "surgery": 0
    }
  },
  "graph": {
    "edges": [
      [
        "pain",
        "LDX4"
      ],
      [
        "LDX4",
        "SZYN"
      ],
      [
        "SZYN",
        "surgery"
      ]
    ],
    "nodes": [
      "pain",
      "LDX4",
      "SZYN",
      "surgery"
    ]
  },
  "id": "ccrgb-clinical-1702-000-05-do1",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "LDX4": 0,
      "SZYN": 1,
      "pain": 0,
      "surgery": 1
    }
  },
  "intervention": {
    "formal": "do(SZYN = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "SZYN",
    "target_span": [
      506,
      510
    ],
    "target_text": "SZYN",
    "text": "Now suppose that lab SZYN will be elevated regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "value": 1,
    "value_token": "1"
  },
  "non_descendants": [
    "LDX4",
    "pain"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "must not change",
      "variable": "pain"
    },
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "must not change",
      "variable": "LDX4"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "SZYN"
    },
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": "Now suppose that lab SZYN will be elevated regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
      "required": "may change",
      "variable": "surgery"
    }
  ],
  "provenance": {
    "converter": "src/convert_ccrgb.py",
    "converter_version": 3,
    "fetched_utc": "2026-09-08T00:00:00Z",
    "file": "ccrgb_6000_worlds.jsonl",
    "sha256": "not-published",
    "url": "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
  },
  "source": "ccrgb",
  "source_id": "1702:0:5:1",
  "structure_kind": "dag"
}
~~~

Assessment: 2 of 4 generated state probes changed and all 2 graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.

### Spot check 4: ccrgb-clinical-2552-000-00-do1

~~~json
{
  "chain": null,
  "descendants": [
    "7HMR",
    "R714",
    "surgery"
  ],
  "factual": {
    "answer": "yes",
    "passage": "Chronic disease KK2JCJ sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 8/10) if they carry allele HFCN, a genetic marker for severe KK2JCJ. If the patient self-reports significant pain or the patient carries allele 7SVC, then lab R714 will be elevated (greater than 2.3 mg/dL). If R714 is elevated or the patient has previously received surgery for UHYO, then lab 7HMR will be elevated (greater than 2.92 mg/dL). If 7HMR is elevated and the patient carries allele OB0T, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here. Now, we will review the history and physical notes for patient Jessica Johnson. History of Present Illness: Jessica Johnson is a 54-year-old female with KK2JCJ who presented to the emergency department with acute onset pain that began 2 hours prior to arrival. Pain was rated 4/10. The patient reports the pain has been persistent since onset. The patient took aspirin (250 mg) at home with minimal relief. Genetic Screening: Patient carries alleles 7SVC, OB0T, MX1L, 3ROE. Family History: 8HQ6, 572F. Medications: QYQ 50 mg/day, FO4 10 mg/day. Past Surgical History: Prior surgeries for I8H6, R3E1.",
    "question": "Given these history and physical notes, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "state": {
      "7HMR": 1,
      "R714": 1,
      "pain": 0,
      "surgery": 1
    }
  },
  "graph": {
    "edges": [
      [
        "pain",
        "R714"
      ],
      [
        "R714",
        "7HMR"
      ],
      [
        "7HMR",
        "surgery"
      ]
    ],
    "nodes": [
      "pain",
      "R714",
      "7HMR",
      "surgery"
    ]
  },
  "id": "ccrgb-clinical-2552-000-00-do1",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "7HMR": 1,
      "R714": 1,
      "pain": 1,
      "surgery": 1
    }
  },
  "intervention": {
    "formal": "do(pain = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "pain",
    "target_span": [
      175,
      179
    ],
    "target_text": "pain",
    "text": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
    "value": 1,
    "value_token": "1"
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "pain"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "R714"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "7HMR"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will the surgeon recommend surgery? Begin your response with Yes or No and be as concise as possible.",
      "required": "may change",
      "variable": "surgery"
    }
  ],
  "provenance": {
    "converter": "src/convert_ccrgb.py",
    "converter_version": 3,
    "fetched_utc": "2026-09-08T00:00:00Z",
    "file": "ccrgb_6000_worlds.jsonl",
    "sha256": "not-published",
    "url": "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
  },
  "source": "ccrgb",
  "source_id": "2552:0:0:1",
  "structure_kind": "dag"
}
~~~

Assessment: 1 of 4 generated state probes changed and all 0 graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.

### Spot check 5: ccrgb-clinical-5267-000-02-do1

~~~json
{
  "chain": null,
  "descendants": [
    "4MML",
    "V4KG",
    "surgery"
  ],
  "factual": {
    "answer": "yes",
    "passage": "Chronic disease 3T9D9H sometimes requires surgical intervention, depending on genetics, patient history, vital signs, and lab results. The patient will experience significant pain (rated greater than or equal to 8/10) if they carry allele CRQK, a genetic marker for severe 3T9D9H. If the patient self-reports significant pain or the patient has previously received surgery for J3SJ, then lab 4MML will be low (less than 3.15 mg/dL). If 4MML is low or the patient has a family history of 823M, then vital V4KG will be elevated (greater than 1.7 mg/dL). If V4KG is elevated and the patient has a family history of YC39, then the surgeon will recommend surgery. Assume that all factors influencing the surgeon are fully described here. Now, we will review the history and physical notes for patient Vanessa Bennett. History of Present Illness: Vanessa Bennett is a 62-year-old female with 3T9D9H who presented to the emergency department with acute onset pain that began 4 hours prior to arrival. Pain was rated 4/10. The patient reports the pain has been persistent since onset. The patient took aspirin (250 mg) at home with minimal relief. Genetic Screening: Patient carries alleles WQBQ, OPUR. Family History: YN41, LBG9. Medications: QJQ 75 mg/day, 1QA 100 mg/day. Past Surgical History: Prior surgeries for J3SJ, 2SUF, GN39.",
    "question": "Given these history and physical notes, will vital V4KG be elevated? Begin your response with Yes or No and be as concise as possible.",
    "state": {
      "4MML": 1,
      "V4KG": 1,
      "pain": 0,
      "surgery": 0
    }
  },
  "graph": {
    "edges": [
      [
        "pain",
        "4MML"
      ],
      [
        "4MML",
        "V4KG"
      ],
      [
        "V4KG",
        "surgery"
      ]
    ],
    "nodes": [
      "pain",
      "4MML",
      "V4KG",
      "surgery"
    ]
  },
  "id": "ccrgb-clinical-5267-000-02-do1",
  "intervened": {
    "answer": "yes",
    "passage": null,
    "state": {
      "4MML": 1,
      "V4KG": 1,
      "pain": 1,
      "surgery": 0
    }
  },
  "intervention": {
    "formal": "do(pain = 1)",
    "kind": "value_set",
    "replacement_span": null,
    "target": "pain",
    "target_span": [
      175,
      179
    ],
    "target_text": "pain",
    "text": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will vital V4KG be elevated? Begin your response with Yes or No and be as concise as possible.",
    "value": 1,
    "value_token": "1"
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": 1,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "pain"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": null,
      "required": "may change",
      "variable": "4MML"
    },
    {
      "actually_changed": false,
      "answer_after": 1,
      "answer_before": 1,
      "question": "Now suppose that the patient will be in significant pain regardless of all other circumstances. With this new assumption, will vital V4KG be elevated? Begin your response with Yes or No and be as concise as possible.",
      "required": "may change",
      "variable": "V4KG"
    },
    {
      "actually_changed": false,
      "answer_after": 0,
      "answer_before": 0,
      "question": null,
      "required": "may change",
      "variable": "surgery"
    }
  ],
  "provenance": {
    "converter": "src/convert_ccrgb.py",
    "converter_version": 3,
    "fetched_utc": "2026-09-08T00:00:00Z",
    "file": "ccrgb_6000_worlds.jsonl",
    "sha256": "not-published",
    "url": "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
  },
  "source": "ccrgb",
  "source_id": "5267:0:2:1",
  "structure_kind": "dag"
}
~~~

Assessment: 1 of 4 generated state probes changed and all 0 graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.
