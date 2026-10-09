# Qwen2.5-32B CLadder operator extension

All 40/40 registered cells completed on validation only. The Qwen2.5-32B-Instruct backbone was frozen and evaluated in BF16 for one-time feature extraction; the canonical O2 and O3 operators and a state head were trained independently for seeds 401–420.

| Measurement | O2 | O3 |
|---|---:|---:|
| Clean composed balanced score | 0.778792 | 0.769169 |
| Pair vectors moved by inversion | 88.780% | 91.236% |
| Inverted score against retained gold | 0.240431 | 0.245068 |
| Do-nothing floor | 0.521163 | 0.510641 |

Paired O2−O3 clean contrast: +0.009623 [+0.002721, +0.016524]. Paired O2−O3 movement contrast: -0.024553 [-0.028871, -0.020235]. These are paired-seed intervals; no held-out test data were read.
