# Experiment data completion

All seven formerly partial or absent experiment data setups now have reproducible generated artifacts. Generated records remain git-ignored; tracked builders and summaries bind their counts and SHA-256 identities.

| Experiment | Status | Concrete artifact |
| --- | --- | --- |
| C1 | Complete | 93,600 records; 31,200 each real, noncausal/placebo, shuffled; 20 graphs |
| C2 | Complete | 7,200 validation floor-suite cases; three executable floors frozen |
| T2 | Complete | 15,000 fresh CSuite development pairs; 15 SEM families; seeds 501-505 |
| T4 | Complete | 45,000 matched CSuite rung views; 15 SEM families; seeds 511-515 |
| T5 | Complete | 5,050 native-task records; 590 generated held-out rows |
| G2 | Complete | 120,000 executable ordered sequences; 1,000 graph-disjoint units |
| G4 | Complete | 4,077 valid doctor annotations; 2,141 adjudicated prompts; 817 articles |

The total newly materialized experiment rows are 289,927, excluding the 2,141 G4 aggregate records and 817 article-manifest rows because those summarize the 4,077 underlying doctor annotations.

No released CSuite or Evidence Inference held-out member was opened. Generated held-out partitions for G2 and T5 exist for later evaluation but were not evaluated.
