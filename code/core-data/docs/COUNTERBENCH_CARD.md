# CounterBench data card

Status: raw source acquired and fully assessed; zero CORE records accepted because the release cannot satisfy the fixed paired-intervention contract without invented data.

## Source and licence

- Pinned source: https://huggingface.co/datasets/CounterBench/CounterBench/tree/not-published
- Licence: `license: mit` (verbatim dataset-card metadata)
- Raw inventory/hashes: `{"data_balanced_alpha_V1.json": "not-published", "data_balanced_backdoor_V2.json": "not-published", "meta_model_alpha_V1.json": "not-published", "meta_model_backdoor_V2.json": "not-published"}`

## Counts and decision

- Raw/candidate questions: 1,200
- Accepted: 0
- Quarantined: 1,200

The natural-language causal clauses can suggest graph arcs, but the release supplies only the counterfactual answer. It does not supply a factual outcome/state pair. Joint and nested rows also contain multiple actions while CORE has one intervention target.

No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.

## Full raw example

~~~json
{
  "question": "Would Lumbo occur if not Ziklo instead of Ziklo?",
  "given_info": "We know that Ziklo causes Blaf, Blaf causes Trune, Trune causes Vork, and Vork causes Lumbo.",
  "answer": "no",
  "type": "basic",
  "question_id": "0",
  "meta": {
    "story_id": "nonsense_0",
    "rung": 3,
    "query_type": "det-counterfactual",
    "model_id": 0,
    "graph_id": "graph5"
  }
}
~~~

## Reproducible quarantine spot checks

There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random(20260903)`; each retains its exact rejection reason so the zero-record decision is auditable.

### Candidate 1

~~~json
{
  "reason": "multiple simultaneous interventions cannot be represented by one intervention.target",
  "record": {
    "answer": "yes",
    "given_info": "We know that Nuv causes Splee, Splee causes Blen, Blen and Splee together cause Druk, Druk causes Plog, not Plog causes Skrim, and Skrim causes Wrox.",
    "meta": {
      "graph_id": "graph7",
      "model_id": 954,
      "query_type": "det-counterfactual",
      "rung": 3,
      "story_id": "nonsense_4"
    },
    "question": "Assume not Nuv, and based on this assumption, further suppose not Druk. Would Wrox occur?",
    "question_id": "990",
    "release": "alpha",
    "type": "nested"
  }
}
~~~

### Candidate 2

~~~json
{
  "reason": "multiple simultaneous interventions cannot be represented by one intervention.target",
  "record": {
    "answer": "yes",
    "given_info": "We know that Fizo causes Blorn, Plim and Quaz, not Quaz causes Skul, Skul causes Triv, Triv causes Yex, Yex causes Rild, and Rild causes Jext.",
    "meta": {
      "graph_id": "graph9",
      "model_id": 553,
      "query_type": "det-counterfactual",
      "rung": 3,
      "story_id": "nonsense_3"
    },
    "question": "Would Jext occur if not Fizo and not Blorn?",
    "question_id": "710",
    "release": "alpha",
    "type": "joint"
  }
}
~~~

### Candidate 3

~~~json
{
  "reason": "counterfactual answer is supplied without a factual outcome/state pair needed to prove an observed change",
  "record": {
    "answer": "yes",
    "given_info": "We know that Nuv causes Splee, Splee and Nuv together cause Blen, Blen causes not Druk, Druk or not Splee causes Plog, Plog causes Skrim, and Skrim causes Wrox. We observed Druk",
    "meta": {
      "graph_id": "graph7",
      "model_id": 319,
      "query_type": "det-counterfactual",
      "rung": 3,
      "story_id": "nonsense_4"
    },
    "question": "Would Wrox occur if not Nuv instead of Nuv?",
    "question_id": "863",
    "release": "alpha",
    "type": "conditional"
  }
}
~~~

### Candidate 4

~~~json
{
  "reason": "counterfactual answer is supplied without a factual outcome/state pair needed to prove an observed change",
  "record": {
    "answer": "no",
    "given_info": "We know that Fizo causes Blorn and Plim, Blorn causes not Quaz, Quaz causes not Skul, Skul causes Triv, Triv causes Jext.",
    "meta": {
      "graph_id": "graph7",
      "model_id": 478,
      "query_type": "det-counterfactual",
      "rung": 3,
      "story_id": "nonsense_3"
    },
    "question": "Would Jext occur if not Fizo instead of Fizo?",
    "question_id": "695",
    "release": "alpha",
    "type": "basic"
  }
}
~~~

### Candidate 5

~~~json
{
  "reason": "multiple simultaneous interventions cannot be represented by one intervention.target",
  "record": {
    "answer": "no",
    "given_info": "We know that Praf causes Vank and Scud, Vank and Scud together cause not Wrenk, Wrenk causes not Yobb and Glim, Glim causes Klep. ",
    "meta": {
      "graph_id": "graph7",
      "model_id": 987,
      "query_type": "det-counterfactual",
      "rung": 3,
      "story_id": "nonsense_2"
    },
    "question": "Assume not Praf, and based on this assumption, further suppose not Wrenk. Would Klep occur?",
    "question_id": "597",
    "release": "alpha",
    "type": "nested"
  }
}
~~~
