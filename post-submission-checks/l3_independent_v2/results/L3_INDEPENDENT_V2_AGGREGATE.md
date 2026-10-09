# Corrected independent L3 expansion

The invalid shared-checkpoint row is superseded by 60 validation-only cells: four frozen backbones, three distinct operator classes, and five seeds per class. All 20 backbone-seed triplets have three distinct checkpoint hashes.

| Method | Balanced accuracy | Identity residual | Idempotence residual | Commutation residual | Last-write residual |
|---|---:|---:|---:|---:|---:|
| `lora_matched` | 0.5342 [0.5240, 0.5444] | 0.2175 [0.1769, 0.2581] | 0.2175 [0.1769, 0.2581] | 0.0000 [0.0000, 0.0000] | 0.2175 [0.1769, 0.2581] |
| `loreft` | 0.5362 [0.5268, 0.5455] | 0.0286 [0.0236, 0.0336] | 0.0285 [0.0234, 0.0335] | 0.0000 [-0.0000, 0.0000] | 0.0285 [0.0235, 0.0335] |
| `o2` | 0.5544 [0.5429, 0.5659] | 0.0041 [0.0025, 0.0057] | 0.0167 [0.0100, 0.0233] | 0.0000 [0.0000, 0.0000] | 0.0167 [0.0100, 0.0233] |

Lower law residual is better. Intervals are Student-t 95% intervals over 20 backbone-seed cells.

Important: `lora_matched` is the corrected frozen-feature parameter-matched adapter. It must not be described as Q/V LoRA on the frozen backbone.
