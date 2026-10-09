# Distributed validation — 2026-09-04

The distributed reader path was validated on the IU small-model cluster server (NVIDIA A16, 15,356 MiB each). These are two-update engineering smokes, not scientific results.

| Reader | GPUs | Fixed examples | Per-GPU batch | Global examples/s | Peak MiB/GPU | Result |
|---|---:|---:|---:|---:|---:|---|
| BERT-base | 4 | 128 | 8 | 73.98 | 2,561 | pass |
| BERT-large | 8 | 256 | 4 | 22.52 | 7,691 | pass |
| BERT-large | 16 | 256 | 2 | 12.81 | 7,691 | pass, slower |

The BERT-large fixed-workload comparison keeps the global batch at 32 for both GPU counts. Eight GPUs delivered 1.76× the throughput of sixteen, so eight is selected. The 16-GPU option remains available for a future operator workload with enough computation to amortize synchronization.

Exact sharded validation returned 128 unique records in all three runs. Rank 0 alone wrote summaries and checkpoints. Held-out test evaluation remained disabled.
