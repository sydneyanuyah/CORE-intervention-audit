# WIQA data card

Status: signed influence-graph conversion complete and verified.

## Source and integrity

- Pinned code revision: `not-published`
- Code snapshot: `data/raw/wiqa/wiqa-edeef924.tar.gz`
- Code snapshot SHA-256: `not-published`
- Influence graphs: 2,107; SHA-256 `not-published`
- Licence: Apache License 2.0
- Primary question release: official no-explanation-v2

- `no_explanation_v2_train.jsonl` (remote `train.jsonl`): `not-published`
- `no_explanation_v2_dev.jsonl` (remote `dev.jsonl`): `not-published`
- `no_explanation_v2_test.jsonl` (remote `test.jsonl`): `not-published`

The primary release contains 39,705 unique questions: 29,808 train, 6,894 development, and 3,003 test. The 2,107 graph rows are required auxiliary metadata; the Hugging Face QA projection alone is insufficient.

## Conversion counts

- Accepted: 26,045
- Quarantined: 13,660
- Accepted splits: 19,758 train, 4,547 validation (source dev), 1,740 test

| Rejection reason | Rows |
|---|---:|
| out-of-paragraph no-effect distractor has no in-graph intervention and no observed change | 13,489 |
| question maps ambiguously to multiple signed graph node pairs | 14 |
| question text does not resolve to a signed graph path | 157 |

The 13,489 out-of-paragraph rows are deliberate no-effect distractors: their source event is normally outside the referenced graph, and they cannot satisfy the required observed-change probe. Another 157 questions do not resolve to a signed path after WIQA's own letters-only normalization; 14 resolve to multiple source/target node pairs and are quarantined rather than guessed.

## Signed graph reconstruction

WIQA graph v1 has fixed signed relations: `V -→ X`, `Z +→ X`, `X -→ W`, `X +→ Y`, and `U -→ Y`. `Y_affects_outcome` determines the signs from `Y` and `W` to acceleration node `A` and deceleration node `D`. Empty grounding groups and their incident edges are removed, matching the pinned source implementation. The fixed CORE edge list cannot carry signs, so `graph.edges` preserves topology while signed path resolution is reflected in directional probe answers and documented here.

## Field mapping

| CORE field | WIQA source or derivation |
|---|---|
| `source_id` | `metadata.ques_id` |
| `graph` | active symbolic v1 nodes and unsigned projection of fixed signed edges |
| `factual.passage` | linked graph `paragraph` preserved exactly as a prefix, followed by deterministic ` [TARGET] <source event>` pointer text |
| `factual.state` | `null`; WIQA encodes directional effects, not world values |
| `factual.question` | `question.stem`, verbatim |
| `factual.answer` | `null`; the source answer is post-change direction |
| intervention target | uniquely resolved structural source graph node; `target_text` is the exact appended source phrase and `target_span` addresses only that phrase |
| intervention value/text | source event phrase parsed verbatim from the stem; `value_token` is the canonical signed-node direction `more` or `less` |
| `intervened.state` | `null` |
| `intervened.answer` | normalized source label `more` or `less` |
| descendants | graph reachability from the source node |
| non-descendants | other active nodes excluding source and descendants |
| probes | queried descendant goes from directional baseline `no_effect` to source answer; non-descendants remain `no_effect` |

## Caveats

- WIQA metrics are direction-of-change accuracy (`more`/`less`/`no_effect`), not state-value matching.
- The probe baseline `no_effect` means no directional change before applying the stated perturbation; it is not a recovered factual state.
- The accepted set excludes all no-effect items because the shared verifier requires at least one observed changed probe.
- Symbolic nodes may have several natural-language groundings. Only questions resolving to exactly one source/target pair at the released path length and answer sign are accepted.
- Most source-event phrases are absent from the original process paragraph. The converter preserves that paragraph byte-for-byte as a prefix and appends one marked source phrase, avoiding fabricated alignment or loss of 25,982 otherwise valid records.
- Official train/dev/test files are preserved; `dev` is named `validation` in CORE manifests.

## Licence text, verbatim

```text
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## First raw question and linked graph, verbatim fields

### Question

```json
{
  "question": {
    "stem": "suppose there will be fewer new trees happens, how will it affect LESS forest formation.",
    "para_steps": [
      "A tree produces seeds",
      "The seeds are dispersed by wind, animals, etc",
      "The seeds reach the ground",
      "Grow into new trees",
      "These new trees produce seeds",
      "The process repeats itself over and over",
      ""
    ],
    "answer_label": "more",
    "answer_label_as_choice": "A",
    "choices": [
      {
        "label": "A",
        "text": "more"
      },
      {
        "label": "B",
        "text": "less"
      },
      {
        "label": "C",
        "text": "no effect"
      }
    ]
  },
  "metadata": {
    "ques_id": "influence_graph:1217:144:106#0",
    "graph_id": "144",
    "para_id": "1217",
    "question_type": "INPARA_EFFECT",
    "path_len": 2
  }
}
```

### Linked influence graph

```json
{
  "para_id": "1217",
  "prompt": "How do forests form?",
  "paragraph": "A tree produces seeds. The seeds are dispersed by wind, animals, etc. The seeds reach the ground. Grow into new trees. These new trees produce seeds. The process repeats itself over and over. ",
  "para_outcome_accelerate": "MORE forest formation?",
  "para_outcome_decelerate": "LESS forest formation",
  "Y_is_outcome": "",
  "X": "more seeds are produced",
  "Y": "more seeds reach the ground",
  "W": [
    "there will be fewer new trees",
    "there will be less habitat for birds"
  ],
  "U": [
    "the trees are felled by humans",
    "the seeds are eaten by animals"
  ],
  "Z": [
    "Excellent weather occurs",
    "there is no human intervention"
  ],
  "V": [
    "there is a widespread  disease among trees",
    "some of the trees are felled"
  ],
  "Y_affects_outcome": "more",
  "graph_id": "144"
}
```

## Reproducible random spot checks

Five accepted records were sampled with Python `random.Random(20260903)`.

### Spot check 1: `wiqa-influence_graph:518:462:42#0`

```json
{
  "chain": null,
  "descendants": [
    "A",
    "D",
    "W",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "An adult fish lays eggs. The eggs incubate for a few weeks or months before hatching. The young fish remain where they hatched until they are big enough to venture into the open water. The fish continue to grow. The fish reach adulthood. Reproduce continuing the cycle. [TARGET] there are less fish that reach adulthood",
    "question": "suppose there are less fish that reach adulthood happens, how will it affect there will be more fish to spawn.",
    "state": null
  },
  "graph": {
    "edges": [
      [
        "V",
        "X"
      ],
      [
        "Z",
        "X"
      ],
      [
        "X",
        "W"
      ],
      [
        "X",
        "Y"
      ],
      [
        "U",
        "Y"
      ],
      [
        "W",
        "A"
      ],
      [
        "W",
        "D"
      ],
      [
        "Y",
        "A"
      ],
      [
        "Y",
        "D"
      ]
    ],
    "nodes": [
      "A",
      "D",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z"
    ]
  },
  "id": "wiqa-influence_graph:518:462:42#0",
  "intervened": {
    "answer": "less",
    "passage": null,
    "state": null
  },
  "intervention": {
    "formal": "directional_change(X = \"there are less fish that reach adulthood\")",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      279,
      319
    ],
    "target_text": "there are less fish that reach adulthood",
    "text": "there are less fish that reach adulthood",
    "value": "there are less fish that reach adulthood",
    "value_token": "more"
  },
  "non_descendants": [
    "U",
    "V",
    "Z"
  ],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "less",
      "answer_before": "no_effect",
      "question": "suppose there are less fish that reach adulthood happens, how will it affect there will be more fish to spawn.",
      "required": "may change",
      "variable": "W"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "U"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "V"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "Z"
    }
  ],
  "provenance": {
    "converter": "src/convert_wiqa.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/wiqa/no_explanation_v2_test.jsonl",
    "sha256": "not-published",
    "url": "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/test.jsonl"
  },
  "source": "wiqa",
  "source_id": "influence_graph:518:462:42#0",
  "structure_kind": "dag"
}
```

Assessment: the queried node is reachable from `X`, its directional answer changes from `no_effect` to `less`, and all 3 non-descendant probes are preserved.

### Spot check 2: `wiqa-influence_graph:1051:1226:110#0`

```json
{
  "chain": null,
  "descendants": [
    "A",
    "D",
    "W",
    "X",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "Iron is exposed to oxygen. Is exposed to air. A chemical reaction occurs. The iron starts to oxidize. The iron starts to rust. [TARGET] there is air exposed on iron",
    "question": "suppose there is air exposed on iron happens, how will it affect there is no air on the iron.",
    "state": null
  },
  "graph": {
    "edges": [
      [
        "V",
        "X"
      ],
      [
        "Z",
        "X"
      ],
      [
        "X",
        "W"
      ],
      [
        "X",
        "Y"
      ],
      [
        "U",
        "Y"
      ],
      [
        "W",
        "A"
      ],
      [
        "W",
        "D"
      ],
      [
        "Y",
        "A"
      ],
      [
        "Y",
        "D"
      ]
    ],
    "nodes": [
      "A",
      "D",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z"
    ]
  },
  "id": "wiqa-influence_graph:1051:1226:110#0",
  "intervened": {
    "answer": "less",
    "passage": null,
    "state": null
  },
  "intervention": {
    "formal": "directional_change(Z = \"there is air exposed on iron\")",
    "kind": "value_set",
    "replacement_span": null,
    "target": "Z",
    "target_span": [
      136,
      164
    ],
    "target_text": "there is air exposed on iron",
    "text": "there is air exposed on iron",
    "value": "there is air exposed on iron",
    "value_token": "more"
  },
  "non_descendants": [
    "U",
    "V"
  ],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "less",
      "answer_before": "no_effect",
      "question": "suppose there is air exposed on iron happens, how will it affect there is no air on the iron.",
      "required": "may change",
      "variable": "W"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "U"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "V"
    }
  ],
  "provenance": {
    "converter": "src/convert_wiqa.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/wiqa/no_explanation_v2_train.jsonl",
    "sha256": "not-published",
    "url": "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/train.jsonl"
  },
  "source": "wiqa",
  "source_id": "influence_graph:1051:1226:110#0",
  "structure_kind": "dag"
}
```

Assessment: the queried node is reachable from `Z`, its directional answer changes from `no_effect` to `less`, and all 2 non-descendant probes are preserved.

### Spot check 3: `wiqa-influence_graph:675:788:123#0`

```json
{
  "chain": null,
  "descendants": [
    "A",
    "D",
    "W",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "Animals and plants die in soft soil or mud. Sediment builds up over the remains. The remains decompose, leaving only trace amounts. The remaining parts over the years are replaced with mineral. A fossil is formed.  [TARGET] more sediment builds up over the remains",
    "question": "suppose more sediment builds up over the remains happens, how will it affect the remains decompose less.",
    "state": null
  },
  "graph": {
    "edges": [
      [
        "V",
        "X"
      ],
      [
        "Z",
        "X"
      ],
      [
        "X",
        "W"
      ],
      [
        "X",
        "Y"
      ],
      [
        "U",
        "Y"
      ],
      [
        "W",
        "A"
      ],
      [
        "W",
        "D"
      ],
      [
        "Y",
        "A"
      ],
      [
        "Y",
        "D"
      ]
    ],
    "nodes": [
      "A",
      "D",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z"
    ]
  },
  "id": "wiqa-influence_graph:675:788:123#0",
  "intervened": {
    "answer": "less",
    "passage": null,
    "state": null
  },
  "intervention": {
    "formal": "directional_change(X = \"more sediment builds up over the remains\")",
    "kind": "value_set",
    "replacement_span": null,
    "target": "X",
    "target_span": [
      224,
      264
    ],
    "target_text": "more sediment builds up over the remains",
    "text": "more sediment builds up over the remains",
    "value": "more sediment builds up over the remains",
    "value_token": "more"
  },
  "non_descendants": [
    "U",
    "V",
    "Z"
  ],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "less",
      "answer_before": "no_effect",
      "question": "suppose more sediment builds up over the remains happens, how will it affect the remains decompose less.",
      "required": "may change",
      "variable": "W"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "U"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "V"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "Z"
    }
  ],
  "provenance": {
    "converter": "src/convert_wiqa.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/wiqa/no_explanation_v2_train.jsonl",
    "sha256": "not-published",
    "url": "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/train.jsonl"
  },
  "source": "wiqa",
  "source_id": "influence_graph:675:788:123#0",
  "structure_kind": "dag"
}
```

Assessment: the queried node is reachable from `X`, its directional answer changes from `no_effect` to `less`, and all 3 non-descendant probes are preserved.

### Spot check 4: `wiqa-influence_graph:51:1068:77#0`

```json
{
  "chain": null,
  "descendants": [
    "A",
    "D",
    "Y"
  ],
  "factual": {
    "answer": null,
    "passage": "Magma moves closer to the Earth&#x27;s crust. The magma starts to cool. The cooling causes atoms in the magma to condense. The condensed magma solidifies. The solidified magma forms minerals. [TARGET] The volcano moves off the hot spot",
    "question": "suppose The volcano moves off the hot spot happens, how will it affect LESS minerals forming.",
    "state": null
  },
  "graph": {
    "edges": [
      [
        "V",
        "X"
      ],
      [
        "Z",
        "X"
      ],
      [
        "X",
        "W"
      ],
      [
        "X",
        "Y"
      ],
      [
        "U",
        "Y"
      ],
      [
        "W",
        "A"
      ],
      [
        "W",
        "D"
      ],
      [
        "Y",
        "A"
      ],
      [
        "Y",
        "D"
      ]
    ],
    "nodes": [
      "A",
      "D",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z"
    ]
  },
  "id": "wiqa-influence_graph:51:1068:77#0",
  "intervened": {
    "answer": "more",
    "passage": null,
    "state": null
  },
  "intervention": {
    "formal": "directional_change(U = \"The volcano moves off the hot spot\")",
    "kind": "value_set",
    "replacement_span": null,
    "target": "U",
    "target_span": [
      201,
      235
    ],
    "target_text": "The volcano moves off the hot spot",
    "text": "The volcano moves off the hot spot",
    "value": "The volcano moves off the hot spot",
    "value_token": "less"
  },
  "non_descendants": [
    "V",
    "W",
    "X",
    "Z"
  ],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "more",
      "answer_before": "no_effect",
      "question": "suppose The volcano moves off the hot spot happens, how will it affect LESS minerals forming.",
      "required": "may change",
      "variable": "D"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "V"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "W"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "X"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "Z"
    }
  ],
  "provenance": {
    "converter": "src/convert_wiqa.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/wiqa/no_explanation_v2_train.jsonl",
    "sha256": "not-published",
    "url": "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/train.jsonl"
  },
  "source": "wiqa",
  "source_id": "influence_graph:51:1068:77#0",
  "structure_kind": "dag"
}
```

Assessment: the queried node is reachable from `U`, its directional answer changes from `no_effect` to `more`, and all 4 non-descendant probes are preserved.

### Spot check 5: `wiqa-influence_graph:1087:1681:133#0`

```json
{
  "chain": null,
  "descendants": [
    "A",
    "D"
  ],
  "factual": {
    "answer": null,
    "passage": "You pack up your car with food, tents, sleeping bags, etc. Drive to your camping spot. Put up your tent. Hide your food from bears. Make a campfire. Roast marshmellows. Go to sleep in the tent. Wake up and enjoy nature.  [TARGET] can not stay and enjoy nature as long",
    "question": "suppose can not stay and enjoy nature as long happens, how will it affect a LESS EXTENSIVE camping trip.",
    "state": null
  },
  "graph": {
    "edges": [
      [
        "V",
        "X"
      ],
      [
        "Z",
        "X"
      ],
      [
        "X",
        "W"
      ],
      [
        "X",
        "Y"
      ],
      [
        "U",
        "Y"
      ],
      [
        "W",
        "A"
      ],
      [
        "W",
        "D"
      ],
      [
        "Y",
        "A"
      ],
      [
        "Y",
        "D"
      ]
    ],
    "nodes": [
      "A",
      "D",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z"
    ]
  },
  "id": "wiqa-influence_graph:1087:1681:133#0",
  "intervened": {
    "answer": "more",
    "passage": null,
    "state": null
  },
  "intervention": {
    "formal": "directional_change(W = \"can not stay and enjoy nature as long\")",
    "kind": "value_set",
    "replacement_span": null,
    "target": "W",
    "target_span": [
      230,
      267
    ],
    "target_text": "can not stay and enjoy nature as long",
    "text": "can not stay and enjoy nature as long",
    "value": "can not stay and enjoy nature as long",
    "value_token": "less"
  },
  "non_descendants": [
    "U",
    "V",
    "X",
    "Y",
    "Z"
  ],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "more",
      "answer_before": "no_effect",
      "question": "suppose can not stay and enjoy nature as long happens, how will it affect a LESS EXTENSIVE camping trip.",
      "required": "may change",
      "variable": "D"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "U"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "V"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "X"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "Y"
    },
    {
      "actually_changed": false,
      "answer_after": "no_effect",
      "answer_before": "no_effect",
      "question": null,
      "required": "must not change",
      "variable": "Z"
    }
  ],
  "provenance": {
    "converter": "src/convert_wiqa.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/wiqa/no_explanation_v2_dev.jsonl",
    "sha256": "not-published",
    "url": "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/dev.jsonl"
  },
  "source": "wiqa",
  "source_id": "influence_graph:1087:1681:133#0",
  "structure_kind": "dag"
}
```

Assessment: the queried node is reachable from `W`, its directional answer changes from `no_effect` to `more`, and all 5 non-descendant probes are preserved.
