# T3 instruction-sensitivity audit

**Disposition:** T3 supplies no paraphrase-robustness result.

## Evidence

| Check | Frozen checkpoints | Matched records | Task predictions changed | Variable predictions changed | Accuracy movement |
|---|---:|---:|---:|---:|---:|
| Clean vs. registered paraphrase | 20 | 2,880 | 0 | 0 | 0.000000 |
| Clean vs. inverted binary value | 20 | 2,880 | 0 | 0 | 0.000000 |

The original paraphrase arm changed command wording but produced bit-identical task and variable decisions for every matched record. A validation-only positive control then reversed each binary instruction value while keeping the target fixed. It also produced bit-identical decisions for every record. Because a meaning-reversing command did not move any decision, zero movement under paraphrase cannot be interpreted as robustness.

The inverted arm intentionally retained the original gold labels. It is a decision-sensitivity control, not a task-accuracy evaluation.

## Provenance

- Protocol: `t3_inverted_instruction_positive_control_v1`
- Manifest: `registry/t3_inverted_control_manifest.json`
- Manifest SHA-256: `not-published`
- Source: frozen A1 CLadder T3-b checkpoints
- Split: validation only
- Distributed world size: four ranks per cell
- Test accessed: no
- Completed cells: 20/20

## Reporting rule

Do not describe T3 as “paraphrase invariant” or as a positive robustness result. The structural-renaming contrast is also not promoted because its paired interval crosses zero and the instruction-sensitivity gate failed.

This disposition is specific to the frozen A1 CLadder T3-b addressing checkpoints. A separately registered follow-up on the F2 CLadder O2/O3 operator artifacts moved 88.8% and 73.8% of pair-level prediction vectors under value inversion. The F2 result therefore must not be voided by extrapolating this A1-path failure.
