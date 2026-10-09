# L2 dropped-law ablation

All 320 registered validation-only cells completed: two families, twenty paired seeds, five learned law profiles, and three deterministic floors. No held-out test was accessed.

The primary estimate is `full − dropped` validation balanced intervention score. Positive values would mean that retaining the named law improved the selected-checkpoint score. Intervals are paired across the twenty registered seeds.

## xor

| Dropped objective | Full mean | Dropped mean | Full − dropped | Student-t 95% | Paired-seed bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| identity | 0.493829 | 0.493809 | +0.000020 | [-0.000009, +0.000050] | [+0.000000, +0.000051] |
| idempotence | 0.493829 | 0.493820 | +0.000010 | [-0.000093, +0.000113] | [-0.000082, +0.000102] |
| commutation | 0.493829 | 0.493829 | +0.000000 | [-0.000031, +0.000031] | [-0.000030, +0.000031] |
| selective_invariance | 0.493829 | 0.493775 | +0.000055 | [-0.000097, +0.000206] | [-0.000063, +0.000204] |

### Full model versus deterministic floors

| Floor | Full − floor | Student-t 95% |
|---|---:|---:|
| do_nothing | -0.006171 | [-0.011864, -0.000477] |
| do_everything | -0.034328 | [-0.040854, -0.027802] |
| random_init | -0.006892 | [-0.012920, -0.000865] |

## cladder

| Dropped objective | Full mean | Dropped mean | Full − dropped | Student-t 95% | Paired-seed bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| identity | 0.513559 | 0.513559 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| idempotence | 0.513559 | 0.513559 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| commutation | 0.513559 | 0.513559 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| selective_invariance | 0.513559 | 0.513559 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |

### Full model versus deterministic floors

| Floor | Full − floor | Student-t 95% |
|---|---:|---:|
| do_nothing | +0.013559 | [-0.002824, +0.029942] |
| do_everything | -0.163761 | [-0.180144, -0.147378] |
| random_init | +0.027581 | [+0.004257, +0.050905] |

## Interpretation

Across the eight family-by-dropped-law primary comparisons, 0 paired-seed Student-t intervals excluded zero. L2 supplies no evidence that retaining any single registered law objective improves the selected-checkpoint balanced score.

Graph-level inference is unavailable because the frozen summaries do not retain per-graph predictions. The report fails closed at paired-seed inference instead of presenting a graph bootstrap that cannot be reconstructed.
