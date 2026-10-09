# CausalT5K data card

Status: raw source acquired and fully assessed; zero CORE records accepted because the release cannot satisfy the fixed paired-intervention contract without invented data.

## Source and licence

- Pinned source: https://github.com/genglongling/CausalT5kBench/tree/not-published
- Licence: Dataset: `CC-BY-4.0`; code: `MIT` (verbatim identifiers from the repository README).
- Raw inventory/hashes: `{"archive_sha256": "not-published"}`

## Counts and decision

- Raw/candidate questions: 7,260
- Accepted: 0
- Quarantined: 7,260

The ten named domains and Pearl levels are preserved in quarantine metadata. `variables` and free-text `causal_structure` do not constitute a complete machine-readable edge list, and no paired world states are released.

No train/validation/test manifests contain IDs because there are no accepted records. Any source-provided split or domain label is retained in the rejected candidate or conversion summary.

## Full raw example

~~~json
{
  "id": "T3-BucketLarge-E-2.001",
  "case_id": "2.001",
  "bucket": "BucketLarge-E",
  "pearl_level": "L1",
  "domain": "Daily Life",
  "subdomain": "Health & Self-Tracking",
  "scenario": "Jordan measures high blood pressure at urgent care after rushing through traffic. Jordan starts an over-the-counter magnesium supplement. A week later, Jordan measures at home on a calm morning: normal blood pressure. Jordan concludes: “Magnesium lowered my blood pressure.”",
  "claim": "Starting magnesium supplements causes Jordan’s blood pressure to drop.",
  "label": "NO",
  "is_ambiguous": false,
  "variables": {
    "X": {
      "name": "Starting magnesium supplement",
      "role": "exposure"
    },
    "Y": {
      "name": "Blood pressure level (or change)",
      "role": "outcome"
    },
    "Z": [
      "Measurement context (stressful urgent care vs. calm home)"
    ]
  },
  "trap": {
    "type": "W4",
    "type_name": "Regression to the Mean",
    "subtype": "Initial_Spike_Regression",
    "subtype_name": "Initial Spike Regression"
  },
  "difficulty": "Easy",
  "causal_structure": "Z → X and Z → Y (an extreme, context-inflated first reading triggers supplement use; repeat readings tend to be lower in calmer contexts)",
  "key_insight": "A single extreme measurement often moves closer to baseline on repeat; changing measurement context can look like a treatment effect.",
  "hidden_timestamp": "Was the first high blood pressure reading taken during an unusually stressful context (e.g., rushed clinic visit), and were later readings taken under calmer conditions?",
  "conditional_answers": {
    "answer_if_condition_1": "NO — the observed drop does not establish that magnesium caused the reduction.",
    "answer_if_condition_2": "PARTIAL/UNCERTAIN — magnesium might help, but you still need controls."
  },
  "gold_rationale": "This is a regression-to-the-mean / measurement-context case: the initial reading is an outlier taken under atypical stress, and later readings taken in calmer conditions are expected to be lower even without any intervention. A single before/after reading isn’t enough. Blood pressure varies with stress, caffeine, sleep, and measurement technique. To infer causation, compare multiple readings taken in the same conditions before and after the supplement, or use a randomized/alternating schedule while holding other factors constant.",
  "initial_author": "Chenyang Dai",
  "validator": "Ryan He",
  "final_score": 10.0,
  "validator_2": "Longling Geng",
  "final_score_2": 10.0
}
~~~

## Reproducible quarantine spot checks

There are no normalized records to spot-check. The following five rejected candidates were sampled with `random.Random(20260903)`; each retains its exact rejection reason so the zero-record decision is auditable.

### Candidate 1

~~~json
{
  "reason": "item has no machine-readable edge list or paired factual/intervened states",
  "record": {
    "annotation": {
      "adjudicated": false,
      "author": "Samantha van Rijs",
      "num_annotators": 1
    },
    "bucket": "BucketLarge-D",
    "case_id": "0102",
    "causal_structure": "Application deterrence -> Stricter eligibility rules, Application deterrence -> Fraud incidence",
    "claim": "The new eligibility rules reduced welfare fraud.",
    "conditional_answers": {
      "answer_if_condition_1": "If Stricter eligibility rules remains a valid proxy for Fraud incidence even after being made a target, the claim holds.",
      "answer_if_condition_2": "If agents are optimizing for Stricter eligibility rules directly without improving the underlying Fraud incidence, then the metric has ceased to be a valid measure."
    },
    "difficulty": "Hard",
    "domain": "Public Policy",
    "final_score": 10.0,
    "final_score_2": 10.0,
    "gold_rationale": "Reducing reported fraud by discouraging applications conflates true fraud reduction with reduced program access, breaking the proxy-target relationship. We cannot definitively conclude that the new eligibility rules reduced welfare fraud because reducing reported fraud by discouraging applications conflates true fraud reduction with reduced program access, breaking the proxy-target relationship. This suggests a potential GOODHART issue.",
    "hidden_timestamp": "Did Application deterrence occur or change before the exposure?",
    "id": "T3-BucketD-0102",
    "initial_author": "Samantha van Rijs",
    "is_ambiguous": false,
    "key_insight": "Reducing reported fraud by discouraging applications conflates true fraud reduction with reduced pro",
    "label": "NO",
    "pearl_level": "L2",
    "release_domain": "D10",
    "release_level": "L2",
    "scenario": "A government introduces stricter eligibility rules for welfare programs to reduce fraud. The number of reported fraud cases declines sharply after implementation. Advocacy groups report that many eligible individuals also stop applying due to increased administrative complexity.",
    "subdomain": "Public Policy",
    "trap": {
      "subtype": "Policy Target Gaming",
      "type": "GOODHART"
    },
    "validator": "Manolo Alvarez",
    "validator_2": "Longling Geng",
    "variables": {
      "X": "Stricter eligibility rules",
      "Y": "Fraud incidence",
      "Z": [
        "Application deterrence"
      ]
    }
  }
}
~~~

### Candidate 2

~~~json
{
  "reason": "item has no machine-readable edge list or paired factual/intervened states",
  "record": {
    "bucket": "BucketLarge-B",
    "case_id": "5.149",
    "causal_structure": "Demographic status (Z) determines both the presence of a landline (X) and the actual inflation experience/expectations (Y). As the population shifts away from landlines, the sample becomes an increasingly biased representation of a shrinking demographic.",
    "claim": "If we extended this landline-based survey for the next five years, it would continue to accurately represent the inflation expectations of the entire population.",
    "conditional_answers": {
      "answer_if_condition_1": "If the whole population used landlines and only recently began to diverge in expectations following the survey implementation, the claim might have been temporarily valid. [N/A]",
      "answer_if_condition_2": "If younger, mobile-only households already faced higher rent/food inflation, then the sampling frame (X) was biased from the start, making the claim [INVALID]."
    },
    "difficulty": "Medium",
    "domain": "Economics",
    "final_score": 10.0,
    "final_score_2": 10.0,
    "gold_rationale": "This is a Selection Bias (F1/T1) error. The survey conditions on landline ownership (X), which is a non-random trait correlated with age and urban stability (Z). Because mobile-only and rural households are systematically excluded, the inference (Y) cannot be generalized to the population, especially as the coverage error worsens over time. The claim is flawed because it ignores structural coverage error. Because landline ownership is caused by demographic traits (Z) that also influence inflation exposure, the survey results are 'lying' by omission of the most vulnerable segments of the economy.",
    "hidden_timestamp": "Did the divergence between mobile-only and landline-using households' consumption baskets emerge before the survey period (X)?",
    "id": "T3-BucketLarge-B-5.149",
    "initial_author": "Vivek Sathe",
    "is_ambiguous": false,
    "key_insight": "Representativeness is not static. If the sampling technology (landlines) is correlated with age and location, the survey ignores the groups most likely to experience volatile price shocks.",
    "label": "NO",
    "pearl_level": "L2",
    "release_domain": "D5",
    "release_level": "L2",
    "scenario": "A central bank runs a household inflation-expectations survey by calling landline numbers in urban areas. The surveyed households report stable expectations around 2%. The bank concludes that 'inflation expectations are well anchored across the population.'",
    "subdomain": "Macroeconomics",
    "trap": {
      "subtype": "T1 - Selection Bias",
      "subtype_name": "Selection Bias",
      "type": "T1",
      "type_name": "Deterministic / Sampling Error"
    },
    "validator": "Vivek Sathe",
    "validator_2": "Longling Geng",
    "variables": {
      "X": {
        "name": "Sampling frame restricted to urban landline owners",
        "role": "exposure"
      },
      "Y": {
        "name": "Accuracy of national inflation expectation inference",
        "role": "outcome"
      },
      "Z": [
        "Demographic shifts toward mobile-only and rural households"
      ]
    }
  }
}
~~~

### Candidate 3

~~~json
{
  "reason": "L1 association item has no intervention",
  "record": {
    "bucket": "BucketLarge-A",
    "case_id": "4.24",
    "causal_structure": "The relationship between birth season and schizophrenia is an aggregate pattern influenced by many correlated social and diagnostic factors, not evidence of a direct causal pathway.",
    "claim": "Prenatal seasonal infections lead to higher schizophrenia risk.",
    "conditional_answers": {
      "answer_if_condition_1": "If seasonal differences reflect changes in diagnosis, reporting, or conception patterns, then birth month is only a proxy and not a cause.",
      "answer_if_condition_2": "If controlled individual-level studies link prenatal infection timing to schizophrenia outcomes, the causal claim would be more plausible."
    },
    "difficulty": "Medium",
    "domain": "Medicine",
    "final_score": 9.0,
    "final_score_2": 9.0,
    "gold_rationale": "A weak seasonal association combined with multiple plausible confounders means the observed pattern cannot justify a specific biological causal explanation. The available data do not support the claim that seasonal infections cause schizophrenia. Aggregate correlations can arise from many confounders and should not be overinterpreted as causal.",
    "hidden_timestamp": "Was prenatal infection directly measured at the individual level before the later onset of schizophrenia, independent of seasonal and social confounders?",
    "id": "T3-BucketLarge-A-4.24-P3-2",
    "initial_author": "Jordan",
    "is_ambiguous": false,
    "key_insight": "Small population-level effects can be misleading when interpreted as evidence for specific individual-level causes.",
    "label": "NO",
    "pearl_level": "L1",
    "release_domain": "D4",
    "release_level": "L1",
    "scenario": "Researchers observe that people born in winter months show a modestly higher incidence of schizophrenia, prompting speculation that maternal infections during winter pregnancies are the cause.",
    "subdomain": "Medicine",
    "trap": {
      "subtype": "Aggregate_Pattern_Overinterpretation",
      "subtype_name": "ECOLOGICAL_FALLACY",
      "type": "W5",
      "type_name": "Ecological Fallacy"
    },
    "validator": "Rebecca Joseph",
    "validator_2": "Longling Geng",
    "variables": {
      "X": {
        "name": "Birth month (winter)",
        "role": "exposure"
      },
      "Y": {
        "name": "Schizophrenia rate",
        "role": "outcome"
      },
      "Z": [
        "Socioeconomic status",
        "Conception timing",
        "Diagnostic variation"
      ]
    }
  }
}
~~~

### Candidate 4

~~~json
{
  "reason": "item has no machine-readable edge list or paired factual/intervened states",
  "record": {
    "annotation": {
      "adjudicated": true,
      "author": "April Yang",
      "num_annotators": 1
    },
    "bucket": "Bucket-Assignment1-F1F",
    "claim": "The sanitation program reduced disease.",
    "conditional_answers": {
      "answer_if_population_shift_occurred": "If the program caused a migration of high-risk individuals into districts with better infrastructure, the aggregate decrease in disease is a statistical artifact of composition, and the program actually increased disease risk for individuals.",
      "answer_if_population_was_static": "If the population distribution remained constant, it would be mathematically impossible for disease rates to rise in every sub-group while falling in the aggregate, suggesting the data provided is inconsistent or missing a third-party confounding variable."
    },
    "correct_answer": "The aggregate reduction came from demographic shifts, not health improvement. Disease worsened in every district; the overall improvement reflects population redistribution toward healthier areas.",
    "detailed_scores": {
      "conditional_answer_a": {
        "justification": "Logically sound; it explains how a composition shift (migration) creates the illusion of success while individual risk increases.",
        "score": 1.5
      },
      "conditional_answer_b": {
        "justification": "Correctly identifies the mathematical impossibility of the scenario under static conditions, though it could more explicitly state that the claim is 'Invalid' in this branch.",
        "score": 1.2
      },
      "difficulty_calibration": {
        "justification": "Medium is appropriate for Simpson's Paradox, as it requires understanding aggregate vs. disaggregate data trends.",
        "score": 1.0
      },
      "final_label": {
        "justification": "The label 'NO' is correct for L2 intervention cases where the causal claim is undermined by statistical paradoxes.",
        "score": 1.0
      },
      "hidden_question_quality": {
        "justification": "The question correctly identifies population distribution/migration as the latent factor that would resolve the paradox.",
        "score": 1.0
      },
      "scenario_clarity": {
        "justification": "The variables X, Y, and Z are clearly defined, and the scenario presents a classic Simpson's Paradox structure in a historical sanitation context.",
        "score": 1.0
      },
      "trap_type": {
        "justification": "Correctly identifies SIMPSON as the trap type, which matches the scenario description.",
        "score": 1.0
      },
      "wise_refusal_quality": {
        "justification": "Follows the template perfectly, identifying the contradictory signals and the need for weighting/migration data.",
        "score": 2.0
      }
    },
    "difficulty": "Hard",
    "domain": "History",
    "final_score": 9.7,
    "final_score_2": 9.7,
    "gold_rationale": "This is Simpson's Paradox. The program may have shifted population toward healthier districts (e.g., slum clearance moved poor residents to middle-class areas), reducing aggregate disease while worsening rates within each district. The compositional change masks universal harm. The claim cannot be definitively evaluated because the aggregate data and the district-level data provide contradictory signals, a classic hallmark of Simpson's Paradox. To determine the true causal effect, we need to know if the sanitation program itself influenced the movement of people between districts, thereby changing the weights of the groups being measured.",
    "hidden_timestamp": "Did the sanitation program cause a shift in population distribution across districts, or were the district populations stable throughout the intervention period?",
    "id": "F1-0129",
    "initial_author": "April Yang",
    "is_ambiguous": false,
    "label": "NO",
    "overall_assessment": "This is a high-quality case that effectively implements Simpson's Paradox in a Pearl Level 2 context. The reasoning is consistent across the hidden question and conditional answers.",
    "pearl_level": "L2",
    "recommendation": "ACCEPT",
    "release_domain": "D2",
    "release_level": "L2",
    "scenario": "A sanitation program (X) reduced disease overall (Y). However, within each district (wealthy, middle, poor), disease rates increased after the program.",
    "trap": {
      "type": "SIMPSON"
    },
    "validator": "April Yang",
    "validator_2": "Longling Geng",
    "variables": {
      "X": "Sanitation program",
      "Y": "Disease rates",
      "Z": [
        "District wealth level",
        "Population distribution across districts"
      ]
    }
  }
}
~~~

### Candidate 5

~~~json
{
  "reason": "item has no machine-readable edge list or paired factual/intervened states",
  "record": {
    "bucket": "BucketLarge-I",
    "case_id": "L2-115",
    "causal_structure": "Selection on extreme leads to regression",
    "claim": "The causal relationship in 'The Fine-Tuning Regression' is valid.",
    "conditional_answers": {
      "condition_A": "If noise present: Smaller improvement is regression, not diminishing returns.",
      "condition_B": "If stable: May be true diminishing returns."
    },
    "difficulty": "Easy",
    "domain": "AI & Tech",
    "final_score": 10.0,
    "final_score_2": 10.0,
    "gold_rationale": "The correct reasoning for this case involves understanding Selection on extreme leads to regression. Best initial loss included favorable noise. This is partly regression to the mean. Runs with best initial loss included favorable noise. Smaller apparent improvement reflects regression, not just diminishing returns.",
    "hidden_timestamp": "Was the best initial loss partly due to favorable noise?",
    "id": "T3-BucketLarge-I-L2-115",
    "initial_author": "Alessandro Balzi",
    "is_ambiguous": false,
    "key_insight": "Best initial loss included favorable noise.",
    "label": "NO",
    "pearl_level": "L2",
    "release_domain": "D8",
    "release_level": "L2",
    "scenario": "Fine-tuning runs with best initial loss (X) show smaller improvements (Y). Team blames diminishing returns.",
    "subdomain": "Transfer Learning",
    "trap": {
      "subtype": "F2_STATISTICAL",
      "subtype_name": "F2 STATISTICAL",
      "type": "T5_REGRESSION",
      "type_name": "T5 Regression"
    },
    "validator": "Alessandro Balzi",
    "validator_2": "Longling Geng",
    "variables": {
      "X": {
        "name": "Best Initial Loss",
        "role": "Selection"
      },
      "Y": {
        "name": "Improvement After Fine-Tuning",
        "role": "Outcome"
      },
      "Z": [
        {
          "name": "Random variance",
          "role": "Source of extreme"
        }
      ]
    }
  }
}
~~~
