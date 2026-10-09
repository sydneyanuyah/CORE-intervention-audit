# T2/T4/G4 Annotation Bible: 300-per-dataset Pilot

Version: 1.0

This Bible governs three separate blind production queues: **300 CausalT5K items, 300 METER items, and 300 PubMedCausal items**. Five additional worked examples from each dataset are excluded from production. The total frozen selection is therefore 900 production items plus 15 teaching items.

The examples below are completed senior-annotator decisions against the current CORE evidence contract. A rejection is a valid annotation result. It means the released row does not itself justify a formal graph/intervention pair; annotators must not fill missing worlds from common sense.

## Frozen sampling contract

- Selection seed: `317` (ordinary three-digit seed).
- CausalT5K: 10 records per D1-D10 x L1-L3 stratum.
- METER: 100 intact contexts x discovery/intervention/counterfactual.
- PubMedCausal: 120 implicit/intra, 120 explicit/intra, 30 implicit/inter, and 30 explicit/inter relation candidates.
- PubMedCausal inputs are train and validation only. The held-out test file is neither read nor sampled.
- Every teaching example is disjoint from its 300-item production queue.
- CausalT5K rows contain independent T2 and T4 annotation forms, so one source review can support both tasks without duplicate labor.

## Senior decision rule

Accept only when the authoritative source identifies the graph (or deterministic SCM), the complete factual state, a surgical intervention, and the complete intervened state. If any element is missing, mark `reject` or `needs_adjudication`, cite the available evidence, and leave unsupported structured fields null. Multiple-choice options, causal wording, and published cause/effect spans are not substitutes for paired worlds.

## CausalT5K: five completed examples

For every example, annotate T2 and T4 separately. T2 asks whether the item can support held-out-domain evaluation. T4 asks whether its released level can support rung-transfer evidence.

### CausalT5K example 1: D1/L1/T3-BucketLarge-E-1.197

Scenario: A survey asks remote workers whether they do “deep work” and whether they are “more productive.” The survey does not define what counts as deep work (e.g., uninterrupted blocks vs any focused time), does not specify the time window (today vs last week), and does not define productivity (tasks completed, quality ratings, or self-perception). The survey summary reports that respondents who say they do deep work also say they are more productive.

Claim: Doing deep work increases productivity among remote workers.

Released causal structure: Because X and Y lack operational definitions and timing, the causal question is not well-posed; it is unclear what intervention or outcome measure the claim refers to.

Completed annotation:

```json
{
  "T2": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "causal_structure",
        "text": "Because X and Y lack operational definitions and timing, the causal question is not well-posed; it is unclear what intervention or outcome measure the claim refers to."
      }
    ],
    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "NO_INTERVENTION"
    ]
  },
  "T4": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "release_level",
        "text": "L1"
      }
    ],
    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "NO_INTERVENTION"
    ]
  }
}
```

Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.

### CausalT5K example 2: D3/L2/T3-BucketD-0042

Scenario: A prestigious venture capital firm publishes a 10-year retrospective highlighting their portfolio companies. The report analyzes 25 'success stories' that achieved valuations over $100 million, showing these companies shared common traits: aggressive growth strategies, charismatic founders, and willingness to operate at losses for market share. The VC firm recommends these traits to new portfolio companies. However, the analysis excludes 150 portfolio companies that failed using identical strategies. The failed companies were quietly written off, sold for pennies, or shut down without public announcements. Of the 175 total companies funded, only 25 survived using these aggressive tactics - …

Claim: Aggressive growth strategies, charismatic founders, and sustained losses cause startup success.

Released causal structure: Only survivors are analyzed (conditioning on Survival=1), making failed companies invisible. True structure shows aggressive tactics create both rare successes and frequent failures.

Completed annotation:

```json
{
  "T2": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "causal_structure",
        "text": "Only survivors are analyzed (conditioning on Survival=1), making failed companies invisible. True structure shows aggressive tactics create both rare successes and frequent failures."
      }
    ],
    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  },
  "T4": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "release_level",
        "text": "L2"
      }
    ],
    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "FACTUAL_STATE_NOT_IDENTIFIED",
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  }
}
```

Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.

### CausalT5K example 3: D5/L3/T3-BucketLarge-B-5.434

Scenario: A startup team of 3 engineers was struggling to meet deadlines. The founder hired an experienced project manager (PM) who implemented a daily 'Scrum' meeting. Within two weeks, the team's output of completed features doubled, even though no other staff were hired and the project requirements stayed the same.

Claim: If the project manager hadn't been hired, the team would not have achieved this increase in output.

Released causal structure: In a small, controlled environment with no other changes (staff, tools, or scope), a sudden and sustained doubling of output (Y) immediately following an intervention (X) provides strong grounds for causal attribution.

Completed annotation:

```json
{
  "T2": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "causal_structure",
        "text": "In a small, controlled environment with no other changes (staff, tools, or scope), a sudden and sustained doubling of output (Y) immediately following an intervention (X) provides strong grounds for causal attribution."
      }
    ],
    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  },
  "T4": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "release_level",
        "text": "L3"
      }
    ],
    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "FACTUAL_STATE_NOT_IDENTIFIED",
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  }
}
```

Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.

### CausalT5K example 4: D7/L2/T3-BucketLarge-C-7.mhgen.T3.5

Scenario: A study of cases where the defendant took the stand (Z) reveals that those who admitted to prior minor crimes (X) were acquitted (Y) more often than those who claimed a perfect past. A defense attorney advises clients that admitting to crimes builds credibility.

Claim: Admitting to prior crimes increases the chance of acquittal.

Released causal structure: Defendants only testify (Z) if the benefit outweighs the risk of X.

Completed annotation:

```json
{
  "T2": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "causal_structure",
        "text": "Defendants only testify (Z) if the benefit outweighs the risk of X."
      }
    ],
    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  },
  "T4": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "release_level",
        "text": "L2"
      }
    ],
    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "FACTUAL_STATE_NOT_IDENTIFIED",
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  }
}
```

Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.

### CausalT5K example 5: D9/L3/T3-BucketLarge-D9-9.535

Scenario: In a major figure skating competition, Skater A performed a technically demanding quadruple jump successfully, earning a significant boost in the technical elements score. Meanwhile, Skater B executed a flawless routine with fewer high-difficulty elements but superior artistry and presentation. The final scores combined technical elements and artistic impression scores to determine the winner. The judges were instructed to weigh jumps heavily, but also to consider overall performance quality. Skater A's quadruple jump was the only such element attempted by any competitor, while Skater B's artistry was widely praised. Ultimately, Skater A won the gold medal by a narrow margin. Invariants: Ju…

Claim: If Skater A had not performed the quadruple jump, would Skater A still have won the gold medal?

Released causal structure: Skater A's quadruple jump causally increases the technical score, which combined with artistry scores determines the final outcome. Skater B's artistry score provides a counterfactual baseline for comparison.

Completed annotation:

```json
{
  "T2": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "causal_structure",
        "text": "Skater A's quadruple jump causally increases the technical score, which combined with artistry scores determines the final outcome. Skater B's artistry score provides a counterfactual baseline for comparison."
      }
    ],
    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  },
  "T4": {
    "decision": "reject",
    "evidence_spans": [
      {
        "field": "release_level",
        "text": "L3"
      }
    ],
    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
    "factual_state": null,
    "graph": null,
    "intervened_state": null,
    "intervention": null,
    "reason_codes": [
      "FACTUAL_STATE_NOT_IDENTIFIED",
      "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
    ]
  }
}
```

Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.

## METER: five completed examples

Each worked example is one intact context with its discovery, intervention, and counterfactual questions. Keeping the triplet together prevents rung leakage.

### METER example 1: context 564

Context: The castles could be inserted into cycler orbits with considerable savings in fuel by performing a series of low thrust maneuvers: The castle would be placed into an interim orbit upon launch, and then use an Earth-swing-by maneuver to boost it into the final cycler orbit. Assuming the use of conventional fuels, it is possible to estimate the fuel required to establish a cycler orbit. In the case of the Aldrin cycler, use of a gravity assist reduces the fuel requirement by about 24.3 metric tons (26.8 short tons), or 15 percent. Other cyclers showed less impressive improvement, due to the shape of their orbits, and when they encounter the Earth. In the case of the VISIT-1 cycler, the benefi…

- discovery: Why do some cyclers not experience a significant reduction in fuel requirements when using gravity assist?
- counterfactual: Had the VISIT-1 cycler's orbit been designed to better exploit Earth's gravity, how would the situation be different?
- intervention: What will be the consequences if engineers develop a new computational model that can design cycler orbits specifically to maximize the fuel-saving benefits of gravity assists?

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "field": "questions",
      "rungs": [
        "discovery",
        "counterfactual",
        "intervention"
      ]
    }
  ],
  "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.

### METER example 2: context 1636

Context: In 1849, there were estimated to be more than 150,000 Canadian horses, and many were exported from Canada annually. Some were shipped to the West Indies, where they possibly contributed to gaited breeds such as the Paso Fino. By the middle of the 19th century, Canadian horses had spread through the northeastern US, where they were used for racing, as roadsters, and, due to their stamina, to pull freight wagons and stagecoaches. Many played a role in the development of other breeds, including the Morgan horse, the American Saddlebred and the Standardbred. Although used extensively in the US, no efforts were made to establish a purebred population, studbook, or breed association in that count…

- discovery: What does the stamina of Canadian horses lead to?
- counterfactual: What change would occur If the Canadian horse's stamina was only effective for short, intense bursts of speed, rather than long-distance endurance?
- intervention: What will happen if a new equestrian endurance race, specifically for purebred Canadian horses, is established along historic northeastern US stagecoach routes?

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "field": "questions",
      "rungs": [
        "discovery",
        "counterfactual",
        "intervention"
      ]
    }
  ],
  "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.

### METER example 3: context 689

Context: Prior to the jamming, the FCC warned that anyone interfering with television signals would be harshly dealt with, and MacDougall was charged after surrendering to the authorities following media and industry pressure. Investigators from the commission spoke to MacDougall in July (he lost his job at Central Florida Teleport beforehand due to the closure of People's Choice), asking him questions that led him to believe that the commission was aware of the incident. Two FCC agents visited MacDougall's house two weeks later along with U.S. Attorney Lawrence Gentile III, who served MacDougall with a subpoena to appear in Jacksonville's U.S. District Court. In their meeting, MacDougall claimed no…

- discovery: What was the consequence of The People’s Choice Network shutting down by July 1986?
- counterfactual: What would be affected assuming the People's Choice Network had only scaled back its operations instead of closing completely?
- intervention: What will be the consequences if the U.S. Attorney's office implements a policy to offer plea deals to all individuals accused of signal interference before a formal indictment?

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "field": "questions",
      "rungs": [
        "discovery",
        "counterfactual",
        "intervention"
      ]
    }
  ],
  "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.

### METER example 4: context 2351

Context: Like the modern killer whale as well as many other extant delphinids, O. citoniensis could have hunted in cooperative pods. In regards to diet, it may have been more similar to the modern false killer whale (Pseudorca crassidens) and pygmy killer whale (Feresa attenuata) in that it was a generalist feeder of squid and large fish. The Orcinus lineage may have fished up the food chain, with the primitive O. citoniensis able to target large fish, and the modern killer whale able to target large whales. Due to this researchers have argued it may have been one of most predatory animals of its region, alongside the orcinine Hemisyntrachelus, and the extinct shark megalodon. However, its significa…

- discovery: What is the effect of Orcinus hunting cooperatively in pods and generally feeding on squid and large fish, with O. citoniensis targeting large fish and modern killer whales targeting large whales?
- counterfactual: What change would occur if O. citoniensis had teeth as formidable as modern killer whales?
- intervention: What will happen if paleontologists discover a fossil of O. citoniensis with the remains of a marine mammal in its stomach?

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "field": "questions",
      "rungs": [
        "discovery",
        "counterfactual",
        "intervention"
      ]
    }
  ],
  "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.

### METER example 5: context 1904

Context: ANTICYCLONE HARTMUT ( dubbed the _ BEAST FROM THE EAST_ ( ) ) was a storm that began on 22 February 2018 , and brought a cold wave to Great Britain and Ireland . Anticyclone Hartmut also brought widespread unusually low temperatures and heavy snowfall to large areas . The cold wave combined with Storm Emma , part of the 2017 – 18 European windstorm season , which made landfall in southwest England and the south of Ireland on 2 March . In contrast to usual winter storms , Hartmut was not formed as a normal low pressure area along the jetstream . The initial event was an Arctic outbreak caused by a disordered polar vortex into Central Europe , transporting not only cold air from Siberia to Eu…

- discovery: What effect did the cold wave combining with Storm Emma have?
- counterfactual: What would have happened if the cold wave had arrived a week after Storm Emma passed?
- intervention: What will be the consequences if the government establishes a new emergency protocol requiring major transportation networks to shut down preemptively based on forecasts of similar combined storm events?

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "field": "questions",
      "rungs": [
        "discovery",
        "counterfactual",
        "intervention"
      ]
    }
  ],
  "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.

## PubMedCausal: five completed examples

The first four examples cover the released implicit/explicit and intra/inter strata; the fifth is a non-relation control. PubMedCausal sentence labels can identify candidates for source-linking, but cannot themselves establish an intervention pair.

### PubMedCausal example 1: train/17137/relation/2

Sentence: From a clinical point of view, findings from the available literature suggest that experimentally induced pain impairs postural control and could potentially increases the risk for falls in patients. Interventions aiming to reduce pain in these patients could lead to preservation or improvement of their balance. On the other hand, the same conclusion cannot be drawn for the effect of experimentally induced pain on kinesthesia and joint position sense due to the limited number of studies showing such an effect

Released relation: cause=`experimentally induced pain`, effect=`impairs postural control`, causality=`Implicit`, sententiality=`Intra`.

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "role": "cause",
      "span": [
        83,
        110
      ],
      "text": "experimentally induced pain"
    },
    {
      "role": "effect",
      "span": [
        111,
        135
      ],
      "text": "impairs postural control"
    }
  ],
  "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "NO_INTERVENTION"
  ]
}
```

Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.

### PubMedCausal example 2: train/25796/relation/1

Sentence: This study offers supportive evidence for a causal association between maternal education and offspring birthweight, highlighting the significance of enhancing maternal education to prevent low birthweight

Released relation: cause=`maternal education`, effect=`offspring birthweight`, causality=`Explicit`, sententiality=`Intra`.

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "role": "cause",
      "span": [
        71,
        89
      ],
      "text": "maternal education"
    },
    {
      "role": "effect",
      "span": [
        94,
        115
      ],
      "text": "offspring birthweight"
    }
  ],
  "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "NO_INTERVENTION"
  ]
}
```

Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.

### PubMedCausal example 3: train/19031/relation/2

Sentence: Different types of parent-child communication have different influencing mechanisms on PTSD and PTG. Therefore, distinct intervention strategies are needed targeted to these two psychological reactions

Released relation: cause=`Different types of parent-child communication have different influencing mechanisms on PTSD and PTG`, effect=`distinct intervention strategies are needed to target these two psychological reactions`, causality=`Implicit`, sententiality=`Inter`.

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "role": "cause",
      "span": [
        0,
        99
      ],
      "text": "Different types of parent-child communication have different influencing mechanisms on PTSD and PTG"
    }
  ],
  "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "NO_INTERVENTION"
  ]
}
```

Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.

### PubMedCausal example 4: train/9666/relation/1

Sentence: Researchers have recently begun to seek cognitive explanations for physical symptoms with no obvious biological cause. Concepts such as somatization, somatosensory amplification, and somatosensory catastrophizing have been invoked to explain these phenomena. Somatosensory amplification occurs when these bodily sensations become stronger and more painful. Somatosensory catastrophizing is the tendency to attribute these bodily sensations to unbearable functional modulation or as signs of serious illness. This causes the sufferer to pay excessive attention to these physical sensations. However, there is no scale for evaluating somatosensory catastrophizing, and there are no standard diagnostic…

Released relation: cause=`Somatosensory catastrophizing`, effect=`the sufferer pays excessive attention to the physical sensations`, causality=`Explicit`, sententiality=`Inter`.

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [
    {
      "role": "cause",
      "span": [
        357,
        386
      ],
      "text": "Somatosensory catastrophizing"
    }
  ],
  "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "NO_INTERVENTION"
  ]
}
```

Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.

### PubMedCausal example 5: train/5312/relation/None

Sentence: Our findings suggest that CSVD may increase AD risk, while specific inflammatory cytokines exhibit differential associations with these conditions. Targeting vascular health and inflammation may offer promising therapeutic avenues for managing neurodegenerative diseases

Released relation: cause=``, effect=``, causality=`None`, sententiality=`None`.

Completed annotation:

```json
{
  "decision": "reject",
  "evidence_spans": [],
  "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
  "factual_state": null,
  "graph": null,
  "intervened_state": null,
  "intervention": null,
  "reason_codes": [
    "GRAPH_NOT_IDENTIFIED"
  ]
}
```

Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.

## Annotator handoff checklist

1. Work only from the assigned `pilot_id`; never replace a sampled item.
2. Keep `graph_group_id` intact and use anonymous three-digit annotator IDs such as `ann-001`.
3. Annotate CausalT5K T2 and T4 independently on the same row.
4. Cite exact evidence spans and locators. Unsupported fields remain null.
5. Do not inspect released answer keys while independently labelling; the production queues omit CausalT5K labels/rationales and METER correct-answer indices.
6. Send `needs_adjudication` rows to a second senior reviewer; do not force an accept/reject guess.
7. Never open or annotate PubMedCausal test data in this phase.
8. Accepted rows still require mechanical schema, graph, span, observed-effect, group-leakage, and provenance validation before they become experimental evidence.

## Files and integrity

- CausalT5K queue SHA-256: `not-published`
- METER queue SHA-256: `not-published`
- PubMedCausal queue SHA-256: `not-published`
- Worked examples SHA-256: `not-published`
- Machine-readable manifest: `reports/annotation_pilot_300_manifest.json`
- Local generated queues: `data/annotation_pilots/` (intentionally ignored by Git because they contain third-party records).
