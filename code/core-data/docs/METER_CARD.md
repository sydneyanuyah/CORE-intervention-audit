# METER data card

Status: raw source acquired and fully assessed; zero CORE records accepted because the release cannot satisfy the fixed paired-intervention contract without invented data.

## Source and licence

- Pinned source: https://github.com/SCUNLP/METER/tree/not-published
- Licence: none stated; the pinned repository contains no licence file and its README is only `# METER`
- Raw inventory/hashes: `{"archive_sha256": "not-published"}`

## Counts and decision

- Raw/candidate questions: 12,445
- Accepted: 0
- Quarantined: 12,445

Each context's causal-discovery, intervention, and counterfactual multiple-choice questions are enumerated. None includes an explicit graph plus paired factual/intervened states.

No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.

## Full raw example

~~~json
{
  "context": "Gaara also appears in light novels from the series. He makes a cameo in Kakashi's Story alongside the other Kage. In Shikamaru Hiden he joins his sister Temari and Naruto in the search for the missing Shikamaru Nara whom Gaara values due to his close relationship with Temari. In Sakura's Hiden, Gaara assists the Konohagakure ninja upon their news that a man resembling Sasuke Uchiha is planning to attack the village. In Konoha Hiden, Gaara visits Konohagakure to see the wedding of Naruto and Hinata Hyuga. He also appears as the protagonist of Gaara Hiden which follows his works as the Fifth Kazekage while dealing with his sister's wedding with Shikamaru.",
  "questions": [
    {
      "ladder": "Causal_Discovery",
      "question": "Why does Gaara join Temari and Naruto in the search for the missing Shikamaru Nara?",
      "answer": 2,
      "option_0": "because of his appearances in the series' light novels",
      "option_1": "because of orders from the Wind Country council",
      "option_2": "his close relationship with Temari",
      "option_3": "because of his search for Shikamaru with Temari",
      "option_4": "because of his hostility toward Temari"
    },
    {
      "ladder": "Intervention",
      "question": "What will happen to cooperation between Gaara and Shikamaru's villages if they establish a formal joint-training program for young ninja?",
      "answer": 0,
      "option_0": "Cooperation between the Sand Village and Shikamaru’s village will strengthen.",
      "option_1": "Cooperation between the Sand Village and Shikamaru’s village will continue through Gaara’s support for Shikamaru-related missions.",
      "option_2": "Cooperation between the Sand Village and Shikamaru’s village will weaken.",
      "option_3": "Cooperation between the Sand Village and Shikamaru’s village will expand into a wider military alliance.",
      "option_4": "Stronger cooperation between the Sand Village and Shikamaru’s village will lead them to establish a formal joint-training program."
    },
    {
      "ladder": "Counterfactual",
      "question": "If Gaara had valued Shikamaru primarily for his strategic skills rather than his bond with Temari, how would his decision to join the search for the missing Shikamaru have been different?",
      "answer": 4,
      "option_0": "He would have likely offered Shikamaru a formal role as a strategic advisor to Sunagakure instead of joining the search.",
      "option_1": "He would have likely declined to assist in the search instead of sending any support.",
      "option_2": "Shikamaru would have likely sharpened his strategic skills further instead of Gaara changing how he responded to the search.",
      "option_3": "He would have likely still appeared alongside the other Kage in Kakashi's Story instead of personally joining the search.",
      "option_4": "He would have likely sent his elite shinobi to assist in the search instead of personally joining."
    }
  ]
}
~~~

## Reproducible quarantine spot checks

There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random(20260903)`; each retains its exact rejection reason so the zero-record decision is auditable.

### Candidate 1

~~~json
{
  "reason": "causal-discovery question has no intervention",
  "record": {
    "answer": 4,
    "context": "Daman did most of its damages on Cikobia island, which has a population of around 120. Damage on Cikobia included extensive damages to houses, school buildings, crops, fruitbearing trees and foliage. Water pipes were damaged by fallen trees as a result of high winds from the storm. Despite initial fears of there being some fatalities on Cikobia, no loss of life was recorded due to the storm. This was because the islands 120 residents had evacuated to caves on Cikobia. The Fijian Dependency of Rotuma experienced a significant amount of rainfall from December 5 until December 7, when Cyclone Daman was located near Rotuma.",
    "context_index": 2542,
    "ladder": "Causal_Discovery",
    "option_0": "All 120 residents had remained in their homes on Cikobia.",
    "option_1": "No loss of life had led all 120 residents to take shelter in caves on Cikobia.",
    "option_2": "Houses and school buildings on Cikobia had suffered extensive damage.",
    "option_3": "Cyclone Daman had weakened before reaching Cikobia.",
    "option_4": "All 120 residents had taken shelter in caves on Cikobia.",
    "question": "Why was no loss of life recorded on Cikobia island due to Cyclone Daman?",
    "question_index": 0
  }
}
~~~

### Candidate 2

~~~json
{
  "reason": "natural-language intervention/counterfactual has no explicit graph or paired world states",
  "record": {
    "answer": 3,
    "context": "During the 20th century, blackbuck numbers declined sharply due to excessive hunting, deforestation and habitat degradation. Some blackbucks are killed illegally especially where the species is sympatric with nilgai.",
    "context_index": 1474,
    "ladder": "Counterfactual",
    "option_0": "A natural expansion of forests and grasslands would have made blackbuck habitats more accessible, leading to the rise of wildlife tourism.",
    "option_1": "The government would have established large, fenced reserves, leading to the genetic isolation of blackbuck populations.",
    "option_2": "Hunters would have shifted their focus exclusively to illegal poaching activities.",
    "option_3": "Local communities would have focused on preserving blackbuck habitats to attract visitors, rather than converting the land for other uses.",
    "option_4": "The blackbuck population would have declined even more rapidly due to the stress caused by increased human presence from tourism.",
    "question": "How would the situation have been different if, early in the 20th century, blackbucks had been prized for wildlife tourism instead of for hunting?",
    "question_index": 2
  }
}
~~~

### Candidate 3

~~~json
{
  "reason": "causal-discovery question has no intervention",
  "record": {
    "answer": 4,
    "context": "Aetosaur material was first described by Swiss paleontologist Louis Agassiz in 1844. He named the genus Stagonolepis from the Lossiemouth Sandstone in Elgin, Scotland, but considered it to be a Devonian fish rather than a Triassic reptile. This may be because he considered the strata to be part of the Old Red Sandstone, and thus Paleozoic in age. Agassiz mistook the osteoderms for large rhomboidal scales, which he thought were arranged in a similar pattern to those of gars. He also thought that these supposed scales were very similar to those of the lobe-finned fish Megalichthys due to their large size.",
    "context_index": 850,
    "ladder": "Causal_Discovery",
    "option_0": "Swiss paleontologist Louis Agassiz decided the strata belonged to the Old Red Sandstone because he had already concluded the fossils were Devonian fish.",
    "option_1": "He first provided a formal description of the aetosaur fossils in 1844.",
    "option_2": "Swiss paleontologist Louis Agassiz postponed formally naming the genus until further fossils could be unearthed from the site.",
    "option_3": "Swiss paleontologist Louis Agassiz classified the genus from the Lossiemouth Sandstone in Elgin, Scotland as a Triassic reptile.",
    "option_4": " Swiss paleontologist Louis Agassiz  considered the genus from the Lossiemouth Sandstone in Elgin, Scotland to be a Devonian fish.",
    "question": "What did Louis Agassiz’s belief that the strata belonged to the Paleozoic Old Red Sandstone lead to?",
    "question_index": 0
  }
}
~~~

### Candidate 4

~~~json
{
  "reason": "natural-language intervention/counterfactual has no explicit graph or paired world states",
  "record": {
    "answer": 4,
    "context": "This type of sundial is known as an equatorial sundial because the plane of the dial face is parallel to the Earth's equatorial plane. Two primary examples of modified equatorial sundials are bowstring equatorial and armillary dials. Both of these modified equatorial sundials can be read year-round on the same surface, regardless of whether the Sun is above or below the equatorial plane whereas a standard equatorial sundial changes sides with each equinox and is virtually unreadable near the equinoxes when the Sun is located on the equatorial plane.",
    "context_index": 3508,
    "ladder": "Intervention",
    "option_0": "A recent surge in customer interest for modified sundials will cause the manufacturer to include the guide to capitalize on the trend.",
    "option_1": "The manufacturer will cease production of standard equatorial sundials due to the increased complexity of the included guide.",
    "option_2": "Customers will be able to use their standard equatorial sundials effectively during the equinoxes by following the new guide.",
    "option_3": "The manufacturer will begin producing bowstring equatorial and armillary dials in greater quantities.",
    "option_4": "Customers will start purchasing the modified sundial models to have a single, functional timepiece for the entire year.",
    "question": "What will happen if a manufacturer of historical instruments begins including a small, supplementary guide with their standard equatorial sundials explaining how to use modified sundials during the equinoxes?",
    "question_index": 1
  }
}
~~~

### Candidate 5

~~~json
{
  "reason": "natural-language intervention/counterfactual has no explicit graph or paired world states",
  "record": {
    "answer": 4,
    "context": "The most severe damage from Hurricane Charley occurred in Charlotte County. In Boca Grande, numerous houses sustained extensive roof damage, while thousands of trees and power lines were uprooted or snapped. In Port Charlotte and Punta Gorda, many buildings, RVs, and mobile homes were completely destroyed, while other buildings suffered roofing damage due to the powerful winds.",
    "context_index": 1274,
    "ladder": "Counterfactual",
    "option_0": "The hurricane would have been downgraded to a tropical storm before making landfall.",
    "option_1": "The powerful winds would have still caused extensive roof damage to many buildings.",
    "option_2": "Buildings would have suffered water damage from leaks instead of roofing damage from wind.",
    "option_3": "The heavy rainfall would have caused the storm to move much more slowly across the county.",
    "option_4": "Residents would have moved their RVs and mobile homes to higher ground ahead of the storm.",
    "question": "What would be different if Hurricane Charley had been a much slower storm, bringing heavy rain instead of powerful winds?",
    "question_index": 2
  }
}
~~~
