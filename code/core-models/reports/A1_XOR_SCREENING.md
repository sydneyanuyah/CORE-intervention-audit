# A1 XOR architecture screening

This report covers the complete 30-cell, five-seed XOR architecture screen. It is not the full A1 family or confirmatory evidence. No test split was accessed.

| Arm | Mean balanced | Delta vs T0 | Student-t 95% CI | Paired-graph bootstrap 95% CI | Target | Pointer top-1 | 2-point rule |
|---|---:|---:|---:|---:|---:|---:|:---:|
| t0_id | 0.5264 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0.7559 | n/a | pass |
| t1_text | 0.5244 | -0.0020 | [-0.0069, +0.0028] | [-0.0044, +0.0005] | 0.7473 | n/a | pass |
| t2b_encode_only | 0.5057 | -0.0207 | [-0.0249, -0.0165] | [-0.0270, -0.0152] | 0.5254 | n/a | fail |
| t2a_naive | 0.4933 | -0.0332 | [-0.0439, -0.0225] | [-0.0376, -0.0294] | 0.3437 | n/a | fail |
| t3b_pointer | 0.5379 | +0.0115 | [+0.0098, +0.0132] | [+0.0083, +0.0151] | 0.8887 | 1.0000 | pass |
| t3a_conditioning | 0.5257 | -0.0008 | [-0.0019, +0.0004] | [-0.0020, +0.0004] | 0.7567 | n/a | pass |

Architecture selected on XOR: **t3b_pointer**.

Student-t intervals use the five paired seed deltas (df=4). The deterministic paired-graph bootstrap resamples the 20 graph units and averages paired arm-minus-T0 differences across the same five seeds (10,000 replicates).

The other 100 registered real-family screening cells remain blocked because the normalized sources do not yet provide authoritative executable two-edit truth. Single-edit outcomes are not combined to fabricate that truth, so confirmatory A1 has not started.
