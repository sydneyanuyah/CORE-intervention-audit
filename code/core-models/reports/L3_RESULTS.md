# L3 eight-method law comparison

All 40 registered XOR cells completed with five graph seeds, four ranks per BERT-base cell, method-matched random-initialization floors, and no test access.

**Interpretation guard.** Identity is non-diagnostic by construction for prompting, task-vector addition, router, O1, O2, and O3: their trained-minus-random estimates are exactly zero with zero-width intervals because the relevant no-op path does not distinguish the fitted and random arms. Prompting is likewise non-diagnostic on all four laws because it has no learned editor to differ from its matched random twin. Only matched LoRA and LoReFT yield non-degenerate identity estimates; neither interval excludes zero. The remaining non-identity estimates below are descriptive paired comparisons, and none has a Student-t interval excluding zero.

| Method | Law | Mean trained − random | Student-t 95% | Graph bootstrap 95% |
|---|---|---:|---:|---:|
| prompting | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| prompting | idempotence | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| prompting | commutation | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| prompting | last_write_wins | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| lora_matched | identity | +0.040000 | [-0.063518, +0.143518] | [-0.020000, +0.108889] |
| lora_matched | idempotence | +0.027611 | [-0.042030, +0.097252] | [-0.014778, +0.075889] |
| lora_matched | commutation | +0.020241 | [-0.030244, +0.070726] | [-0.012296, +0.052639] |
| lora_matched | last_write_wins | +0.027574 | [-0.041977, +0.097125] | [-0.014963, +0.075704] |
| loreft | identity | +0.002222 | [-0.003948, +0.008392] | [+0.000000, +0.006667] |
| loreft | idempotence | +0.011611 | [-0.016537, +0.039759] | [-0.002222, +0.031685] |
| loreft | commutation | +0.007389 | [-0.017023, +0.031801] | [-0.004630, +0.024778] |
| loreft | last_write_wins | +0.011407 | [-0.016271, +0.039086] | [-0.002111, +0.031185] |
| task_vector_add | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| task_vector_add | idempotence | -0.002185 | [-0.007632, +0.003262] | [-0.006185, +0.000000] |
| task_vector_add | commutation | +0.000630 | [-0.000441, +0.001700] | [+0.000000, +0.001259] |
| task_vector_add | last_write_wins | -0.001333 | [-0.004221, +0.001554] | [-0.003481, +0.000000] |
| router | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| router | idempotence | +0.000889 | [-0.002120, +0.003897] | [-0.000444, +0.003111] |
| router | commutation | -0.001296 | [-0.004436, +0.001843] | [-0.003611, +0.000000] |
| router | last_write_wins | -0.000556 | [-0.001802, +0.000691] | [-0.001482, +0.000000] |
| o1 | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| o1 | idempotence | -0.002185 | [-0.007632, +0.003262] | [-0.006185, +0.000000] |
| o1 | commutation | +0.000574 | [-0.000410, +0.001558] | [+0.000000, +0.001204] |
| o1 | last_write_wins | -0.001185 | [-0.003774, +0.001404] | [-0.003111, +0.000000] |
| o2 | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| o2 | idempotence | -0.001944 | [-0.007028, +0.003139] | [-0.005648, +0.000000] |
| o2 | commutation | +0.000019 | [-0.000187, +0.000224] | [-0.000111, +0.000167] |
| o2 | last_write_wins | -0.000815 | [-0.002421, +0.000791] | [-0.002000, +0.000000] |
| o3 | identity | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| o3 | idempotence | +0.001074 | [-0.003756, +0.005904] | [-0.001444, +0.004667] |
| o3 | commutation | +0.000176 | [-0.004163, +0.004515] | [-0.002694, +0.003222] |
| o3 | last_write_wins | -0.001167 | [-0.003184, +0.000850] | [-0.002500, +0.000000] |
