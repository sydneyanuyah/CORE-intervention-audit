# Com2 data card

Status: conversion complete and verified.

## Source and integrity

- Pinned revision: `not-published`
- Benchmark URL: `https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json`
- Local raw file: `data/raw/com2/main.json`
- Bytes: 9,128,252
- SHA-256: `not-published`
- Licence: none stated. The pinned repository root has no licence file and its README contains no licence statement.

The raw file contains 2,500 objects: 500 each of `counterfactual`, `decision`, `direct`, `intervention`, and `reco`. This differs from the task note, which called the first type `transition`; the converter asserts the observed release rather than renaming it.

## Selection and counts

| Output | Accepted | Rejected | Chain prefix findings |
|---|---:|---:|---|
| `com2_intervention.jsonl` | 270 | 0 | `k=1` for all 270; 258 chains are 5/5 and 12 are 6/6 |
| `com2_counterfactual.jsonl` | 500 | 0 | `k=0` for 448 and `k=1` for 52; lengths are 5/4 for 4, 5/5 for 495, and 6/6 for 1 |

The other 230 intervention rows have no `chains` key and are not converted. `direct`, `decision`, and `reco` are outside this converter's specified selection. Empty rejection files are emitted so a later failure cannot be confused with an unrecorded omission.

## Field mapping

| CORE field | Com2 source or derivation |
|---|---|
| `id` | `com2-<type>-<zero-padded original list index>` |
| `source_id` | zero-based index in untouched `main.json` |
| `structure_kind` | `chain` |
| `graph` | `null`; Com2 supplies paths, not a DAG |
| `chain` | factual path `chains[0]`, verbatim |
| `factual.passage` | source `scenario` preserved as a prefix, followed by marked `[ORIG]` and `[REPL]` event candidates for pointer addressing |
| `factual.question` | `question`, verbatim |
| `factual.answer` | `null`; the only source `answer` answers the intervention/counterfactual question |
| `factual.state` | `null`; no variable-state map is supplied |
| intervention target/value | first divergent position `k`: the position-qualified before event is the structural target; exact `target_span` and `replacement_span` address the marked before/after event text; `value_token` is `null` |
| `intervention.text` | source `question`, verbatim |
| `intervened.answer` | source `answer`, verbatim |
| intervened passage/state | the same marked model-input passage / `null` state |
| `non_descendants` | position-qualified occurrences before `k` |
| `descendants` | position-qualified factual occurrences after `k` |
| probes | aligned before/after occurrences; prefix is `must not change`, target and suffix are `may change` |

Event occurrences are labelled `event_NN: <verbatim event>` because eight source chains repeat identical event text at different positions. This disambiguates identity without changing `chain`, which remains verbatim. For the four 5/4 counterfactual pairs, the unmatched final factual event remains a descendant but has no probe because the source gives no after value.

## Splits

Com2 provides no authoritative train/validation/test assignment for these rows. Exact factual root-event groups are assigned together using `sha256('com2-split-v1:' + root_event) mod 100`: buckets 0–79 train, 80–89 validation, and 90–99 test. This prevents the same exact root event from crossing splits.

- Intervention: 223 train, 24 validation, 23 test
- Counterfactual: 392 train, 49 validation, 59 test

## Caveats

- These are chain/path records. Descendant labels mean later positions on a supplied path, not graph reachability.
- `[ORIG]` and `[REPL]` are deterministic non-asserted pointer candidates appended to the source scenario; both character spans are validated as exact slices.
- The 448 counterfactual rows with `k=0` have no non-descendants and therefore cannot test preservation.
- A `may change` probe is allowed to remain unchanged. The target always differs, satisfying the observed-effect requirement.
- Answers are multiple-choice strings, while probes compare event strings. They measure different views of the same intervention.
- No redistribution licence is stated; the raw data should not be redistributed without permission from its owners.

## Full raw item, verbatim fields

The first selected intervention item (original zero-based index 2) is reproduced below without field changes:

```json
{
  "question": "If I enroll in a language immersion program but do not improve my language skills, what is the outcome I should expect in terms of my academic journey?",
  "options": "A) Make new friends or professional connections  \nB) Write a thesis that gets published  \nC) Develop a passion for linguistics  \nD) Gain confidence in speaking",
  "thinking_process": "1. Context Identification: I recognize that I am evaluating the outcomes of enrolling in a language immersion program but facing an obstacle where I do not improve my language skills. This indicates a disruption in the first causal chain, as it prevents the chain from proceeding to its intended outcome.  \n2. Divide: I need to break down the causal relationships incorporated in the options. The first chain outlines the progression of gaining social connections through language skills, while the second chain contains outcomes related to academics.  \n3. Comparison: I will go through each option to see how they relate to the outcomes that stem from the language immersion program, noting that only one can be valid based on the intervention:  \n   - Option A (Make new friends or professional connections) relies directly on improved skills, which I don’t have.  \n   - Option B (Write a thesis that gets published) comes from a path that assumes progress in academics, starting after developing a passion.  \n   - Option C (Develop a passion for linguistics) could potentially happen, as it doesn't directly depend on language skills; the passion may manifest differently.  \n   - Option D (Gain confidence in speaking) is certainly invalid since I am not improving my language skills.  \n4. Self-Refinement: I need to assess my reasoning here; if my language skills don't improve, my potential for making friends and gaining confidence diminishes significantly. Option C remains appealing since it doesn’t assert a dependency on improved language skills.  \n5. Conquer: After evaluating options based on the causal structure initiated, I see that while A and D are ruled out, option C could still lead to the unpredictable desire for deeper academic pursuit in linguistics, which is a clear progression in the second causal chain.",
  "answer": "C) Develop a passion for linguistics.",
  "chains": [
    [
      "Enroll in a language immersion program",
      "Improve language skills",
      "Gain confidence in speaking",
      "Use language in social or professional settings",
      "Make new friends or professional connections"
    ],
    [
      "Enroll in a language immersion program",
      "Develop a passion for linguistics",
      "Pursue a master's degree in linguistics",
      "Write a thesis that gets published",
      "Become a renowned academic in the field of linguistics"
    ]
  ],
  "scenario": "Maria had always dreamed of becoming fluent in Spanish, so she decided to enroll in a summer language immersion program in Spain. Each day, she attended intensive language classes and engaged in conversational practice with locals. As weeks passed, she noticed a significant improvement in her language skills, which boosted her confidence in speaking. \n\nEncouraged by her progress, Maria started using Spanish in social situations, chatting with her classmates and striking up conversations with shopkeepers and café owners. Through these interactions, she made several new friends, including a group of local university students who invited her to join them for weekend outings. \n\nBy the end of the program, not only had Maria improved her Spanish, but she had also built a network of personal and professional connections, sparking a newfound passion for the language and culture that she knew would last a lifetime.",
  "type": "intervention",
  "model": "gpt-4o-mini"
}
```

## Reproducible random spot checks

Five records were sampled from the 770 accepted rows with Python `random.Random(20260903)`. Each full normalized record follows.

### Spot check 1: `com2-counterfactual-000941`

```json
{
  "chain": [
    "Rearrange your bookshelf",
    "Discover a book you forgot you owned",
    "Read the book during your free time",
    "Develop a new interest in a different genre",
    "Join a book club to discuss your newfound passion"
  ],
  "descendants": [
    "event_01: Discover a book you forgot you owned",
    "event_02: Read the book during your free time",
    "event_03: Develop a new interest in a different genre",
    "event_04: Join a book club to discuss your newfound passion"
  ],
  "factual": {
    "answer": null,
    "passage": "After moving into a new apartment, Sarah decides to take the time to rearrange her bookshelf, which has been a chaotic mix of genres and authors since her college days. As she organizes the books by category, she stumbles upon a dusty copy of \"The Night Circus,\" a novel she had completely forgotten about. Intrigued, she picks it up and starts reading it during her weekend downtime. The enchanting story captivates her, sparking a newfound interest in fantasy literature, a genre she had never explored before. Excited by her discovery, Sarah joins a local book club that focuses on fantasy and magical realism, where she shares her thoughts and connects with others who share her passion for reading.\n[ORIG] Rearrange your bookshelf\n[REPL] Counterfactual_of_given_event (You were never able to rearrange your bookshelf)",
    "question": "If you were never able to rearrange your bookshelf, what will happen?",
    "state": null
  },
  "graph": null,
  "id": "com2-counterfactual-000941",
  "intervened": {
    "answer": "B) You will develop a narrow worldview, limiting personal and educational growth.",
    "passage": "After moving into a new apartment, Sarah decides to take the time to rearrange her bookshelf, which has been a chaotic mix of genres and authors since her college days. As she organizes the books by category, she stumbles upon a dusty copy of \"The Night Circus,\" a novel she had completely forgotten about. Intrigued, she picks it up and starts reading it during her weekend downtime. The enchanting story captivates her, sparking a newfound interest in fantasy literature, a genre she had never explored before. Excited by her discovery, Sarah joins a local book club that focuses on fantasy and magical realism, where she shares her thoughts and connects with others who share her passion for reading.\n[ORIG] Rearrange your bookshelf\n[REPL] Counterfactual_of_given_event (You were never able to rearrange your bookshelf)",
    "state": null
  },
  "intervention": {
    "formal": "do(event_00 = \"Counterfactual_of_given_event (You were never able to rearrange your bookshelf)\")",
    "kind": "event_replace",
    "replacement_span": [
      743,
      822
    ],
    "target": "event_00: Rearrange your bookshelf",
    "target_span": [
      711,
      735
    ],
    "target_text": "Rearrange your bookshelf",
    "text": "Replace Rearrange your bookshelf with Counterfactual_of_given_event (You were never able to rearrange your bookshelf).",
    "value": "Counterfactual_of_given_event (You were never able to rearrange your bookshelf)",
    "value_token": null
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "Counterfactual_of_given_event (You were never able to rearrange your bookshelf)",
      "answer_before": "Rearrange your bookshelf",
      "question": null,
      "required": "may change",
      "variable": "event_00: Rearrange your bookshelf"
    },
    {
      "actually_changed": true,
      "answer_after": "Realize your collection is disorganized and unwieldy",
      "answer_before": "Discover a book you forgot you owned",
      "question": null,
      "required": "may change",
      "variable": "event_01: Discover a book you forgot you owned"
    },
    {
      "actually_changed": true,
      "answer_after": "Avoid reading altogether due to frustration",
      "answer_before": "Read the book during your free time",
      "question": null,
      "required": "may change",
      "variable": "event_02: Read the book during your free time"
    },
    {
      "actually_changed": true,
      "answer_after": "Miss out on new perspectives and knowledge",
      "answer_before": "Develop a new interest in a different genre",
      "question": null,
      "required": "may change",
      "variable": "event_03: Develop a new interest in a different genre"
    },
    {
      "actually_changed": true,
      "answer_after": "Develop a narrow worldview, limiting personal and educational growth",
      "answer_before": "Join a book club to discuss your newfound passion",
      "question": null,
      "required": "may change",
      "variable": "event_04: Join a book club to discuss your newfound passion"
    }
  ],
  "provenance": {
    "converter": "src/convert_com2.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/com2/main.json",
    "sha256": "not-published",
    "url": "https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json"
  },
  "source": "com2",
  "source_id": 941,
  "structure_kind": "chain"
}
```

Assessment: the target is the first divergent occurrence; all 0 preservation probes agree before/after, and 5 of 5 may-change probes differ. The normalized mapping is structurally consistent with the two source paths.

### Spot check 2: `com2-counterfactual-000022`

```json
{
  "chain": [
    "explore new technology",
    "develop a prototype",
    "conduct market research",
    "launch a product",
    "gain customer feedback"
  ],
  "descendants": [
    "event_01: develop a prototype",
    "event_02: conduct market research",
    "event_03: launch a product",
    "event_04: gain customer feedback"
  ],
  "factual": {
    "answer": null,
    "passage": "In a bustling tech startup located in Silicon Valley, a team of engineers and designers gathers to explore the latest advancements in artificial intelligence. They brainstorm potential applications and decide to focus on developing an innovative personal assistant app that can help users manage their daily tasks. Excited by their idea, they quickly move into the prototyping phase, creating a functional model that showcases the app's capabilities.\n\nAfter refining their prototype, the team conducts thorough market research, surveying potential users and analyzing competitors to identify gaps in the market. Armed with valuable insights, they iterate on their design and prepare for the official launch of the product, which they anticipate will change the way people interact with technology.\n\nThe launch day arrives, and they unveil the app at a major tech conference, generating buzz and attracting a significant number of downloads. Customers flock to the app store, eager to try it out. In the following weeks, they actively gather feedback from users through in-app surveys and social media channels, allowing them to understand customer satisfaction and identify areas for improvement. This feedback loop helps the team enhance the app further, ensuring it meets users' evolving needs and solidifying its place in the market.\n[ORIG] explore new technology\n[REPL] explore new technology in a world without electricity",
    "question": "If a personal assistant app is developed in a world without electricity, what will the outcome be?",
    "state": null
  },
  "graph": null,
  "id": "com2-counterfactual-000022",
  "intervened": {
    "answer": "D) The invention faces challenges in widespread adoption due to logistical constraints.",
    "passage": "In a bustling tech startup located in Silicon Valley, a team of engineers and designers gathers to explore the latest advancements in artificial intelligence. They brainstorm potential applications and decide to focus on developing an innovative personal assistant app that can help users manage their daily tasks. Excited by their idea, they quickly move into the prototyping phase, creating a functional model that showcases the app's capabilities.\n\nAfter refining their prototype, the team conducts thorough market research, surveying potential users and analyzing competitors to identify gaps in the market. Armed with valuable insights, they iterate on their design and prepare for the official launch of the product, which they anticipate will change the way people interact with technology.\n\nThe launch day arrives, and they unveil the app at a major tech conference, generating buzz and attracting a significant number of downloads. Customers flock to the app store, eager to try it out. In the following weeks, they actively gather feedback from users through in-app surveys and social media channels, allowing them to understand customer satisfaction and identify areas for improvement. This feedback loop helps the team enhance the app further, ensuring it meets users' evolving needs and solidifying its place in the market.\n[ORIG] explore new technology\n[REPL] explore new technology in a world without electricity",
    "state": null
  },
  "intervention": {
    "formal": "do(event_00 = \"explore new technology in a world without electricity\")",
    "kind": "event_replace",
    "replacement_span": [
      1374,
      1427
    ],
    "target": "event_00: explore new technology",
    "target_span": [
      1344,
      1366
    ],
    "target_text": "explore new technology",
    "text": "Replace explore new technology with explore new technology in a world without electricity.",
    "value": "explore new technology in a world without electricity",
    "value_token": null
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "explore new technology in a world without electricity",
      "answer_before": "explore new technology",
      "question": null,
      "required": "may change",
      "variable": "event_00: explore new technology"
    },
    {
      "actually_changed": true,
      "answer_after": "develop a mechanical version",
      "answer_before": "develop a prototype",
      "question": null,
      "required": "may change",
      "variable": "event_01: develop a prototype"
    },
    {
      "actually_changed": true,
      "answer_after": "conduct experiments with limited resources",
      "answer_before": "conduct market research",
      "question": null,
      "required": "may change",
      "variable": "event_02: conduct market research"
    },
    {
      "actually_changed": true,
      "answer_after": "implement the invention in daily life",
      "answer_before": "launch a product",
      "question": null,
      "required": "may change",
      "variable": "event_03: launch a product"
    },
    {
      "actually_changed": true,
      "answer_after": "face challenges in widespread adoption due to logistical constraints",
      "answer_before": "gain customer feedback",
      "question": null,
      "required": "may change",
      "variable": "event_04: gain customer feedback"
    }
  ],
  "provenance": {
    "converter": "src/convert_com2.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/com2/main.json",
    "sha256": "not-published",
    "url": "https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json"
  },
  "source": "com2",
  "source_id": 22,
  "structure_kind": "chain"
}
```

Assessment: the target is the first divergent occurrence; all 0 preservation probes agree before/after, and 5 of 5 may-change probes differ. The normalized mapping is structurally consistent with the two source paths.

### Spot check 3: `com2-intervention-001442`

```json
{
  "chain": [
    "try indoor trampoline park fitness",
    "feel motivated to stay active",
    "search for more fitness classes",
    "join a local fitness community",
    "improve overall health and fitness"
  ],
  "descendants": [
    "event_02: search for more fitness classes",
    "event_03: join a local fitness community",
    "event_04: improve overall health and fitness"
  ],
  "factual": {
    "answer": null,
    "passage": "After a long week of work, Sarah decided to try out an indoor trampoline park fitness class that her friend had been raving about. As she bounced around, feeling the exhilaration of jumping and the rush of endorphins, she realized how much fun it was to be active. This experience ignited a spark of motivation within her to stay active and explore more fitness options. \n\nInspired, Sarah began searching online for additional fitness classes in her area and discovered a variety of options, from yoga to kickboxing. She signed up for a few classes and quickly found herself enjoying the camaraderie and support of fellow fitness enthusiasts. This led her to join a local fitness community that organized group workouts and social events. \n\nAs she became more involved, Sarah noticed significant improvements in her overall health and fitness. She felt stronger, more energetic, and more confident in her body. The journey from that first trampoline class to becoming an active member of a fitness community transformed her lifestyle, making health and fitness a central part of her daily routine.\n[ORIG] feel motivated to stay active\n[REPL] experience a minor injury while jumping",
    "question": "What happens after trying indoor trampoline park fitness if I take a break from physical activities?",
    "state": null
  },
  "graph": null,
  "id": "com2-intervention-001442",
  "intervened": {
    "answer": "D) Become interested in physical therapy and recovery.",
    "passage": "After a long week of work, Sarah decided to try out an indoor trampoline park fitness class that her friend had been raving about. As she bounced around, feeling the exhilaration of jumping and the rush of endorphins, she realized how much fun it was to be active. This experience ignited a spark of motivation within her to stay active and explore more fitness options. \n\nInspired, Sarah began searching online for additional fitness classes in her area and discovered a variety of options, from yoga to kickboxing. She signed up for a few classes and quickly found herself enjoying the camaraderie and support of fellow fitness enthusiasts. This led her to join a local fitness community that organized group workouts and social events. \n\nAs she became more involved, Sarah noticed significant improvements in her overall health and fitness. She felt stronger, more energetic, and more confident in her body. The journey from that first trampoline class to becoming an active member of a fitness community transformed her lifestyle, making health and fitness a central part of her daily routine.\n[ORIG] feel motivated to stay active\n[REPL] experience a minor injury while jumping",
    "state": null
  },
  "intervention": {
    "formal": "do(event_01 = \"experience a minor injury while jumping\")",
    "kind": "event_replace",
    "replacement_span": [
      1142,
      1181
    ],
    "target": "event_01: feel motivated to stay active",
    "target_span": [
      1105,
      1134
    ],
    "target_text": "feel motivated to stay active",
    "text": "Replace feel motivated to stay active with experience a minor injury while jumping.",
    "value": "experience a minor injury while jumping",
    "value_token": null
  },
  "non_descendants": [
    "event_00: try indoor trampoline park fitness"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": "try indoor trampoline park fitness",
      "answer_before": "try indoor trampoline park fitness",
      "question": null,
      "required": "must not change",
      "variable": "event_00: try indoor trampoline park fitness"
    },
    {
      "actually_changed": true,
      "answer_after": "experience a minor injury while jumping",
      "answer_before": "feel motivated to stay active",
      "question": null,
      "required": "may change",
      "variable": "event_01: feel motivated to stay active"
    },
    {
      "actually_changed": true,
      "answer_after": "take a break from physical activities",
      "answer_before": "search for more fitness classes",
      "question": null,
      "required": "may change",
      "variable": "event_02: search for more fitness classes"
    },
    {
      "actually_changed": true,
      "answer_after": "become interested in physical therapy and recovery",
      "answer_before": "join a local fitness community",
      "question": null,
      "required": "may change",
      "variable": "event_03: join a local fitness community"
    },
    {
      "actually_changed": false,
      "answer_after": "improve overall health and fitness",
      "answer_before": "improve overall health and fitness",
      "question": null,
      "required": "may change",
      "variable": "event_04: improve overall health and fitness"
    }
  ],
  "provenance": {
    "converter": "src/convert_com2.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/com2/main.json",
    "sha256": "not-published",
    "url": "https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json"
  },
  "source": "com2",
  "source_id": 1442,
  "structure_kind": "chain"
}
```

Assessment: the target is the first divergent occurrence; all 1 preservation probes agree before/after, and 3 of 4 may-change probes differ. The normalized mapping is structurally consistent with the two source paths.

### Spot check 4: `com2-counterfactual-001958`

```json
{
  "chain": [
    "recover from surgery",
    "rest and follow post-operative instructions",
    "reduce risk of complications",
    "body heals and repairs tissues",
    "regain strength and mobility"
  ],
  "descendants": [
    "event_01: rest and follow post-operative instructions",
    "event_02: reduce risk of complications",
    "event_03: body heals and repairs tissues",
    "event_04: regain strength and mobility"
  ],
  "factual": {
    "answer": null,
    "passage": "After undergoing knee surgery to repair a torn ligament, Sarah returned home feeling relieved but aware that her recovery depended on her actions in the coming weeks. She made a commitment to rest and diligently followed her doctor's post-operative instructions, which included taking prescribed pain medications and attending physical therapy sessions. By adhering to these guidelines, she significantly reduced her risk of complications, such as infection or improper healing.\n\nDuring her recovery period, Sarah focused on maintaining a healthy diet and drinking plenty of water, which aided her body in healing and repairing the tissues around her knee. Gradually, she noticed improvements in her mobility as the swelling subsided and her pain diminished. After a few weeks of consistent effort in her physical therapy, she began to regain strength and mobility in her knee, allowing her to return to her favorite activities like jogging and hiking, feeling grateful for her successful recovery.\n[ORIG] recover from surgery\n[REPL] recover from surgery in a world without pain relief",
    "question": "If Sarah recovers from surgery in a world without pain relief, what is the most likely outcome?",
    "state": null
  },
  "graph": null,
  "id": "com2-counterfactual-001958",
  "intervened": {
    "answer": "B) Sarah may experience prolonged recovery and possible lasting disability.",
    "passage": "After undergoing knee surgery to repair a torn ligament, Sarah returned home feeling relieved but aware that her recovery depended on her actions in the coming weeks. She made a commitment to rest and diligently followed her doctor's post-operative instructions, which included taking prescribed pain medications and attending physical therapy sessions. By adhering to these guidelines, she significantly reduced her risk of complications, such as infection or improper healing.\n\nDuring her recovery period, Sarah focused on maintaining a healthy diet and drinking plenty of water, which aided her body in healing and repairing the tissues around her knee. Gradually, she noticed improvements in her mobility as the swelling subsided and her pain diminished. After a few weeks of consistent effort in her physical therapy, she began to regain strength and mobility in her knee, allowing her to return to her favorite activities like jogging and hiking, feeling grateful for her successful recovery.\n[ORIG] recover from surgery\n[REPL] recover from surgery in a world without pain relief",
    "state": null
  },
  "intervention": {
    "formal": "do(event_00 = \"recover from surgery in a world without pain relief\")",
    "kind": "event_replace",
    "replacement_span": [
      1034,
      1085
    ],
    "target": "event_00: recover from surgery",
    "target_span": [
      1006,
      1026
    ],
    "target_text": "recover from surgery",
    "text": "Replace recover from surgery with recover from surgery in a world without pain relief.",
    "value": "recover from surgery in a world without pain relief",
    "value_token": null
  },
  "non_descendants": [],
  "probes": [
    {
      "actually_changed": true,
      "answer_after": "recover from surgery in a world without pain relief",
      "answer_before": "recover from surgery",
      "question": null,
      "required": "may change",
      "variable": "event_00: recover from surgery"
    },
    {
      "actually_changed": true,
      "answer_after": "experience severe pain and discomfort",
      "answer_before": "rest and follow post-operative instructions",
      "question": null,
      "required": "may change",
      "variable": "event_01: rest and follow post-operative instructions"
    },
    {
      "actually_changed": true,
      "answer_after": "avoid movement and physical therapy",
      "answer_before": "reduce risk of complications",
      "question": null,
      "required": "may change",
      "variable": "event_02: reduce risk of complications"
    },
    {
      "actually_changed": true,
      "answer_after": "increased risk of complications",
      "answer_before": "body heals and repairs tissues",
      "question": null,
      "required": "may change",
      "variable": "event_03: body heals and repairs tissues"
    },
    {
      "actually_changed": true,
      "answer_after": "extended recovery time and possible lasting disability",
      "answer_before": "regain strength and mobility",
      "question": null,
      "required": "may change",
      "variable": "event_04: regain strength and mobility"
    }
  ],
  "provenance": {
    "converter": "src/convert_com2.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/com2/main.json",
    "sha256": "not-published",
    "url": "https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json"
  },
  "source": "com2",
  "source_id": 1958,
  "structure_kind": "chain"
}
```

Assessment: the target is the first divergent occurrence; all 0 preservation probes agree before/after, and 5 of 5 may-change probes differ. The normalized mapping is structurally consistent with the two source paths.

### Spot check 5: `com2-intervention-002216`

```json
{
  "chain": [
    "Sketch a scene",
    "Share the sketch with friends",
    "Friends provide feedback",
    "Friends encourage further practice",
    "Artist improves their skills over time."
  ],
  "descendants": [
    "event_02: Friends provide feedback",
    "event_03: Friends encourage further practice",
    "event_04: Artist improves their skills over time."
  ],
  "factual": {
    "answer": null,
    "passage": "Emily, a budding artist, spends her Saturday afternoon sketching a vibrant street scene filled with bustling cafes and colorful storefronts. Once she finishes, she takes a photo of her sketch and shares it in a group chat with her friends, who are also art enthusiasts. Her friends respond with enthusiastic feedback, praising her use of color and perspective while also suggesting a few techniques to enhance her work. Encouraged by their support, Emily decides to practice more regularly, experimenting with different styles and subjects. Over the next few months, she dedicates time each week to sketching and painting, gradually noticing significant improvements in her skills and confidence as an artist.\n[ORIG] Share the sketch with friends\n[REPL] Submit the sketch to a national competition",
    "question": "Emily sketches a scene and then receives feedback from her friends. What is the likely outcome if she submits her sketch to a national competition instead of practicing further?",
    "state": null
  },
  "graph": null,
  "id": "com2-intervention-002216",
  "intervened": {
    "answer": "D) Emily receives unexpected recognition.",
    "passage": "Emily, a budding artist, spends her Saturday afternoon sketching a vibrant street scene filled with bustling cafes and colorful storefronts. Once she finishes, she takes a photo of her sketch and shares it in a group chat with her friends, who are also art enthusiasts. Her friends respond with enthusiastic feedback, praising her use of color and perspective while also suggesting a few techniques to enhance her work. Encouraged by their support, Emily decides to practice more regularly, experimenting with different styles and subjects. Over the next few months, she dedicates time each week to sketching and painting, gradually noticing significant improvements in her skills and confidence as an artist.\n[ORIG] Share the sketch with friends\n[REPL] Submit the sketch to a national competition",
    "state": null
  },
  "intervention": {
    "formal": "do(event_01 = \"Submit the sketch to a national competition\")",
    "kind": "event_replace",
    "replacement_span": [
      754,
      797
    ],
    "target": "event_01: Share the sketch with friends",
    "target_span": [
      717,
      746
    ],
    "target_text": "Share the sketch with friends",
    "text": "Replace Share the sketch with friends with Submit the sketch to a national competition.",
    "value": "Submit the sketch to a national competition",
    "value_token": null
  },
  "non_descendants": [
    "event_00: Sketch a scene"
  ],
  "probes": [
    {
      "actually_changed": false,
      "answer_after": "Sketch a scene",
      "answer_before": "Sketch a scene",
      "question": null,
      "required": "must not change",
      "variable": "event_00: Sketch a scene"
    },
    {
      "actually_changed": true,
      "answer_after": "Submit the sketch to a national competition",
      "answer_before": "Share the sketch with friends",
      "question": null,
      "required": "may change",
      "variable": "event_01: Share the sketch with friends"
    },
    {
      "actually_changed": true,
      "answer_after": "Receive unexpected recognition",
      "answer_before": "Friends provide feedback",
      "question": null,
      "required": "may change",
      "variable": "event_02: Friends provide feedback"
    },
    {
      "actually_changed": true,
      "answer_after": "Get an offer for a solo art exhibition",
      "answer_before": "Friends encourage further practice",
      "question": null,
      "required": "may change",
      "variable": "event_03: Friends encourage further practice"
    },
    {
      "actually_changed": true,
      "answer_after": "Transition into a full-time artist career.",
      "answer_before": "Artist improves their skills over time.",
      "question": null,
      "required": "may change",
      "variable": "event_04: Artist improves their skills over time."
    }
  ],
  "provenance": {
    "converter": "src/convert_com2.py",
    "converter_version": 2,
    "fetched_utc": "2026-09-03T18:10:38Z",
    "file": "data/raw/com2/main.json",
    "sha256": "not-published",
    "url": "https://raw.githubusercontent.com/Waste-Wood/Com2/not-published/benckmark/com2/main.json"
  },
  "source": "com2",
  "source_id": 2216,
  "structure_kind": "chain"
}
```

Assessment: the target is the first divergent occurrence; all 1 preservation probes agree before/after, and 4 of 4 may-change probes differ. The normalized mapping is structurally consistent with the two source paths.
