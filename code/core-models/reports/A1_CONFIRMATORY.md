# A1 preregistered confirmatory evidence

All 520 registered BERT-base confirmatory cells are complete. Every cell used exactly four GPUs, remained validation-only, and reports `test_evaluated=false`. The architecture was frozen before this matrix; these results do not reselect it.

| Family | Arm | Mean balanced | 95% t interval | Change | Preservation | Target | Unseen paraphrase | Pointer top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xor | t0_id | 0.5211 | [0.5197, 0.5225] | 0.5458 | 0.4964 | 0.7556 | n/a | n/a |
| xor | t1_text | 0.5187 | [0.5174, 0.5201] | 0.5435 | 0.4940 | 0.7542 | 0.9994 | n/a |
| xor | t2b_encode_only | 0.5046 | [0.5015, 0.5077] | 0.4258 | 0.5835 | 0.5012 | 0.9838 | n/a |
| xor | t2a_naive | 0.4929 | [0.4883, 0.4974] | 0.4045 | 0.5812 | 0.3576 | 0.7488 | n/a |
| xor | t3b_pointer | 0.5315 | [0.5295, 0.5336] | 0.5656 | 0.4975 | 0.8951 | 1.0000 | 1.0000 |
| xor | t3a_conditioning | 0.5224 | [0.5207, 0.5241] | 0.5455 | 0.4992 | 0.7595 | 1.0000 | n/a |
| ccrgb | t1_text | 0.5016 | [0.4983, 0.5049] | 0.0077 | 0.9955 | 0.0083 | 0.9125 | n/a |
| ccrgb | t2b_encode_only | 0.2991 | [0.2697, 0.3286] | 0.2846 | 0.3136 | 0.3083 | 0.8250 | n/a |
| ccrgb | t2a_naive | 0.3173 | [0.2696, 0.3651] | 0.2846 | 0.3500 | 0.3083 | 0.7833 | n/a |
| ccrgb | t3b_pointer | 0.5031 | [0.4966, 0.5097] | 0.0154 | 0.9909 | 0.0167 | 0.9167 | 1.0000 |
| ccrgb | t3a_conditioning | 0.5059 | [0.4539, 0.5580] | 0.2096 | 0.8023 | 0.2146 | 0.7333 | n/a |
| cladder | t1_text | 0.5006 | [0.4942, 0.5070] | 0.4937 | 0.5075 | 0.4988 | 0.5173 | n/a |
| cladder | t2b_encode_only | 0.5205 | [0.5147, 0.5263] | 0.5333 | 0.5077 | 0.5051 | 0.5160 | n/a |
| cladder | t2a_naive | 0.4057 | [0.3955, 0.4159] | 0.3862 | 0.4252 | 0.3463 | 0.5238 | n/a |
| cladder | t3b_pointer | 0.4990 | [0.4933, 0.5048] | 0.4864 | 0.5117 | 0.4961 | 0.5150 | 0.2636 |
| cladder | t3a_conditioning | 0.5055 | [0.4996, 0.5114] | 0.4922 | 0.5188 | 0.5023 | 0.5187 | n/a |
| wiqa | t1_text | 0.5565 | [0.5452, 0.5678] | 0.3819 | 0.7311 | 0.3400 | 0.4708 | n/a |
| wiqa | t2b_encode_only | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | 0.4749 | n/a |
| wiqa | t2a_naive | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | 0.4737 | n/a |
| wiqa | t3b_pointer | 0.7022 | [0.6879, 0.7165] | 0.4251 | 0.9792 | 0.2300 | 0.7648 | 0.9661 |
| wiqa | t3a_conditioning | 0.4873 | [0.4631, 0.5116] | 0.3170 | 0.6576 | 0.1771 | 0.7234 | n/a |
| com2 | t1_text | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| com2 | t2b_encode_only | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| com2 | t2a_naive | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| com2 | t3b_pointer | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| com2 | t3a_conditioning | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |

Frozen architecture: **t3b_pointer**. Its confirmatory XOR mean is 0.5315; the paired mean delta versus T0 is +0.0104.

95% intervals for that delta are [+0.0077, +0.0132] by Student t across seeds and [+0.0077, +0.0131] by deterministic paired bootstrap over graphs. The preregistered within-two-points result is **PASS**.

T0 is the synthetic learned-ID reference and remains undefined on real families. No real-family T0 value was fabricated, and no held-out test record was opened.

The open-text rows are excluded from scientific interpretation. Their pointer top-1 value was computed over non-empty address-supervision rows (73 eligible validation records per inspected source summary), while every task and variable score was zero. It is therefore not an empty-denominator artifact, but it is not evidence of usable task performance and is omitted from the research report's A1 table.
