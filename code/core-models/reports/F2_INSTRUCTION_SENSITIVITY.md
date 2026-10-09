# F2 CLadder instruction-sensitivity control

**Disposition:** The F2 O2 and O3 operator paths are decision-sensitive to the requested binary value on validation CLadder.

## Result

Both commands in every registered composed pair were inverted while retaining the original gold outputs only as a sensitivity reference. The exact frozen readers and fitted operators from F2/C3 were reused; no checkpoint was retrained or mutated.

## Mechanism result

O2 is above O3 on both axes of the diagnostic: clean score (0.7562 versus 0.6270) and inversion-induced pair movement (88.8% versus 73.8%). Within this controlled O2/O3 comparison, state gating attenuates command following and provides a mechanism for O3's lower task score. The accompanying figure is `figures/F2_COMMAND_SENSITIVITY.svg`.

| Method | Cells | Pair measurements | Pair vectors changed | Variable decisions changed | Clean balanced score | Inverted score against retained gold | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| O2 | 20 | 12,300 | 10,920 (88.8%) | 30,142/46,260 (65.2%) | 0.7562 | 0.2385 | -0.5177 |
| O3 | 20 | 12,300 | 9,074 (73.8%) | 26,092/46,260 (56.4%) | 0.6270 | 0.3543 | -0.2727 |

The operator path is not command-inert. Reversing the requested values causes large, directionally adverse movement against the original gold in both methods. This distinguishes F2 from the A1 CLadder encode-only path, where the same control changed zero decisions.

This result does not make C3 independent of F2. C3 uses the same full O3 cells and should be presented as F2's editor-zeroed ablation arm. C4 is parameter accounting derived from the same fitted cells.

## Provenance

- Protocol: `f2_cladder_inverted_instruction_positive_control_v1`
- Manifest: `registry/f2_cladder_instruction_sensitivity_manifest.json`
- Manifest SHA-256: `not-published`
- Machine-readable aggregate: `reports/F2_INSTRUCTION_SENSITIVITY.json`
- Source: exact frozen F2 CLadder readers and O2/O3 operator artifacts
- Split: validation only
- Distributed world size: four ranks per cell
- Completed cells: 40/40
- Test accessed: no
