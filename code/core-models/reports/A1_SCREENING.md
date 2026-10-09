# A1 architecture screening — all families

All 130 registered BERT-base architecture-screening cells are complete. Every cell used exactly four GPUs, remained validation-only, and accessed no held-out test records. These five-seed results select architecture; they are not confirmatory evidence.

| Family | Arm | Mean balanced | 95% t interval | Change | Preservation | Target | Pointer top-1 | Rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xor | t0_id | 0.5264 | [0.5259, 0.5269] | 0.5540 | 0.4989 | 0.7559 | n/a | 2 |
| xor | t1_text | 0.5244 | [0.5193, 0.5295] | 0.5491 | 0.4997 | 0.7473 | n/a | 4 |
| xor | t2b_encode_only | 0.5057 | [0.5014, 0.5100] | 0.4400 | 0.5714 | 0.5254 | n/a | 5 |
| xor | t2a_naive | 0.4933 | [0.4830, 0.5035] | 0.3626 | 0.6239 | 0.3437 | n/a | 6 |
| xor | t3b_pointer | 0.5379 | [0.5366, 0.5393] | 0.5748 | 0.5011 | 0.8887 | 1.0000 | 1 |
| xor | t3a_conditioning | 0.5257 | [0.5250, 0.5263] | 0.5518 | 0.4995 | 0.7567 | n/a | 3 |
| ccrgb | t1_text | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | n/a | 2 |
| ccrgb | t2b_encode_only | 0.2720 | [0.2411, 0.3029] | 0.3077 | 0.2364 | 0.3333 | n/a | 5 |
| ccrgb | t2a_naive | 0.2811 | [0.2559, 0.3064] | 0.3077 | 0.2545 | 0.3333 | n/a | 4 |
| ccrgb | t3b_pointer | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 3 |
| ccrgb | t3a_conditioning | 0.5168 | [0.3658, 0.6677] | 0.2154 | 0.8182 | 0.2167 | n/a | 1 |
| cladder | t1_text | 0.4969 | [0.4823, 0.5115] | 0.4773 | 0.5165 | 0.4910 | n/a | 5 |
| cladder | t2b_encode_only | 0.5171 | [0.5077, 0.5264] | 0.5413 | 0.4929 | 0.5087 | n/a | 1 |
| cladder | t2a_naive | 0.5014 | [0.4868, 0.5159] | 0.5189 | 0.4838 | 0.5071 | n/a | 4 |
| cladder | t3b_pointer | 0.5034 | [0.4913, 0.5155] | 0.4750 | 0.5317 | 0.4920 | 0.2667 | 3 |
| cladder | t3a_conditioning | 0.5074 | [0.4698, 0.5449] | 0.4840 | 0.5308 | 0.5037 | n/a | 2 |
| wiqa | t1_text | 0.5383 | [0.5108, 0.5657] | 0.3670 | 0.7095 | 0.3700 | n/a | 2 |
| wiqa | t2b_encode_only | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | n/a | 3 |
| wiqa | t2a_naive | 0.5000 | [0.5000, 0.5000] | 0.0000 | 1.0000 | 0.0000 | n/a | 4 |
| wiqa | t3b_pointer | 0.7192 | [0.6803, 0.7580] | 0.4562 | 0.9821 | 0.4100 | 0.9700 | 1 |
| wiqa | t3a_conditioning | 0.4894 | [0.3956, 0.5832] | 0.4124 | 0.5663 | 0.2871 | n/a | 5 |
| com2 | t1_text | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | n/a | 1 |
| com2 | t2b_encode_only | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | n/a | 2 |
| com2 | t2a_naive | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | n/a | 3 |
| com2 | t3b_pointer | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 4 |
| com2 | t3a_conditioning | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | n/a | 5 |

Frozen architecture selection: **t3b_pointer**. On XOR it scores 0.5379, +0.0115 versus T0, and therefore passes the preregistered two-point rule.

T0 is a synthetic learned-ID reference and is deliberately undefined for the four real families. Real-family results are reported as architecture-screening diagnostics and are not compared to a fabricated T0.
