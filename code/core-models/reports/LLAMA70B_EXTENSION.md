# Llama-3.3-70B CLadder operator extension

All 40/40 registered cells completed on validation only. The Llama-3.3-70B-Instruct backbone was frozen and evaluated in BF16 for one-time feature extraction; the canonical O2 and O3 operators and a state head were trained independently for seeds 401–420.

| Measurement | O2 | O3 |
|---|---:|---:|
| Clean composed balanced score | 0.784847 | 0.760290 |
| Pair vectors moved by inversion | 88.780% | 90.065% |
| Inverted score against retained gold | 0.249246 | 0.285376 |
| Do-nothing floor | 0.527902 | 0.504136 |

Paired O2−O3 clean contrast: +0.024557 [+0.015914, +0.033199]. Paired O2−O3 movement contrast: -0.012846 [-0.016328, -0.009363]. These are paired-seed intervals; no held-out test data were read.
