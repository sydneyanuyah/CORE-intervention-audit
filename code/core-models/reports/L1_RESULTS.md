# L1 Law-Retention Results

All 240 registered cells passed provenance validation; no held-out test rows were evaluated.

| Family | Method | Balanced intervention | Change | Preservation | Variable accuracy |
|---|---|---:|---:|---:|---:|
| xor | core_base_1law | 0.490608 | 0.494570 | 0.486646 | 0.489292 |
| xor | core_i_3law | 0.490598 | 0.494570 | 0.486626 | 0.489278 |
| xor | core_full_4law | 0.490598 | 0.494570 | 0.486626 | 0.489278 |
| xor | do_nothing | 0.500000 | 0.000000 | 1.000000 | 0.694750 |
| xor | do_everything | 0.528157 | 0.552512 | 0.503803 | 0.518556 |
| xor | random_init | 0.500722 | 0.498926 | 0.502517 | 0.501597 |
| cladder | core_base_1law | 0.518680 | 0.506140 | 0.531220 | 0.517686 |
| cladder | core_i_3law | 0.518680 | 0.506140 | 0.531220 | 0.517686 |
| cladder | core_full_4law | 0.518680 | 0.506140 | 0.531220 | 0.517686 |
| cladder | do_nothing | 0.500000 | 0.000000 | 1.000000 | 0.391969 |
| cladder | do_everything | 0.677320 | 0.852201 | 0.502439 | 0.715105 |
| cladder | random_init | 0.485978 | 0.483176 | 0.488780 | 0.485373 |

## Paired retention comparisons

### xor

| Comparison | Mean delta | Student-t 95% | Paired-seed bootstrap 95% |
|---|---:|---:|---:|
| core_i_3law_minus_core_base_1law | -0.000010 | [-0.000031, +0.000011] | [-0.000030, +0.000000] |
| core_full_4law_minus_core_i_3law | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| core_full_4law_minus_core_base_1law | -0.000010 | [-0.000031, +0.000011] | [-0.000030, +0.000000] |
| core_full_4law_minus_do_nothing | -0.009402 | [-0.017114, -0.001690] | [-0.016133, -0.002174] |
| core_full_4law_minus_do_everything | -0.037560 | [-0.044122, -0.030997] | [-0.043097, -0.031271] |
| core_full_4law_minus_random_init | -0.010124 | [-0.019407, -0.000840] | [-0.018630, -0.001383] |

### cladder

| Comparison | Mean delta | Student-t 95% | Paired-seed bootstrap 95% |
|---|---:|---:|---:|
| core_i_3law_minus_core_base_1law | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| core_full_4law_minus_core_i_3law | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| core_full_4law_minus_core_base_1law | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| core_full_4law_minus_do_nothing | +0.018680 | [+0.003007, +0.034353] | [+0.003561, +0.030531] |
| core_full_4law_minus_do_everything | -0.158640 | [-0.174313, -0.142967] | [-0.173662, -0.146477] |
| core_full_4law_minus_random_init | +0.032702 | [+0.010472, +0.054931] | [+0.012769, +0.052998] |

## Inference scope

The preregistered paired-seed Student-t intervals and a paired-seed bootstrap are reported. Graph-level bootstrap inference is unavailable because the frozen cell summaries do not retain per-graph predictions; the reporter fails closed rather than inventing graph-level evidence.
