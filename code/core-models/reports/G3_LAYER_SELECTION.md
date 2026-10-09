# G3 Development Layer Selection Record — Scientific Claim Retired

G3 is complete at 11/11 provenance-valid BERT-base cells. The preregistered deterministic rule mechanically selected **edit layer 4** at validation two-edit balanced accuracy **0.539092**. This selection is retained as an execution record only: it is not a scientific result, and the layer is **not frozen as an evidence-backed downstream choice**.

## Recorded development selection

- Stage: development-only selection
- Seed: `2026090600`
- Candidate layers: 1 through 11
- Mechanically selected layer: **4**
- Selection rule: maximize validation `two_edit_balanced`; break exact ties in favor of the smallest layer
- Development graphs: frozen A1 screening XOR graphs 3001–3020
- Excluded graphs: A1 confirmatory XOR graphs 3021–3040
- Model and allocation: BERT-base, exactly four GPUs per cell
- Completed cells: 11/11, with 11 distinct checkpoint SHA-256 values
- Held-out test accessed: **no** (`test_evaluated=false` for every cell)

## Ranking

| Rank | Layer | Validation two-edit balanced |
|---:|---:|---:|
| 1 | 4 | 0.539092 |
| 2 | 2 | 0.538511 |
| 3 | 7 | 0.538511 |
| 4 | 8 | 0.538511 |
| 5 | 9 | 0.538511 |
| 6 | 10 | 0.538511 |
| 7 | 1 | 0.538031 |
| 8 | 5 | 0.537977 |
| 9 | 11 | 0.537409 |
| 10 | 6 | 0.536045 |
| 11 | 3 | 0.535556 |

## Provenance

- Preregistered manifest commit: `bf896de`
- Launch record commit: `a0e55d1`
- Manifest SHA-256: `not-published`
- small-model cluster-host generated report SHA-256: `not-published`
- Tracked canonical JSON SHA-256: `not-published` (semantically identical; formatting normalized)

The sweep is non-diagnostic for a layer claim. Layers 2, 7, 8, 9, and 10 tied exactly at `0.538511`, and the winning margin over that five-way tie was only `0.000581`. The historical choice of layer 4 is therefore retired as scientific evidence and must not be described as a validated or frozen optimum.
