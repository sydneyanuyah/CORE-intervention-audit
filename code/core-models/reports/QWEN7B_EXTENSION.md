# Qwen2.5-7B CLadder operator extension

All 40/40 registered cells completed on validation only. The Qwen2.5-7B-Instruct backbone was frozen and quantized only for one-time feature extraction; the canonical O2 and O3 operators and a state head were trained independently for seeds 401–420.

| Measurement | O2 | O3 |
|---|---:|---:|
| Clean composed balanced score | 0.770467 | 0.742728 |
| Pair vectors moved by inversion | 88.780% | 89.366% |
| Inverted score against retained gold | 0.238257 | 0.264195 |
| Do-nothing floor | 0.499022 | 0.507884 |

Paired O2−O3 clean contrast: +0.027739 [+0.020687, +0.034790]. Paired O2−O3 movement contrast: -0.005854 [-0.010868, -0.000840]. These are paired-seed intervals; no held-out test data were read.
