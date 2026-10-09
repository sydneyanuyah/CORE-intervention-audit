# Qwen2.5-7B derived C3, C4, and T1 results

These are prespecified derived views of the same 40 F2 extension cells, not independent replications.

| Experiment | Result | Interpretation |
|---|---|---|
| C3 | O3 clean minus editor-bypassed/do-nothing: **+0.234844**, paired-seed 95% CI **[+0.224827,+0.244862]** | The fitted O3 path materially changes the 7B probe's CLadder decisions. |
| C4 | O2: **154,112** operator parameters; O3: **25,905,152**; O3/O2 = **168.09×** | O2 is both more accurate and far smaller in this 7B probe. |
| T1 | O2 composed score **0.770467**; O3 **0.742728** over 615 validation pairs | Composition view of F2; O2−O3 **+0.027739** [**+0.020687,+0.034790**]. |

All source cells use seeds 401–420, validation only, with `test_evaluated=false`.

