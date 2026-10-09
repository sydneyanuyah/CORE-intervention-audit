# A2 frozen-checkpoint measurement

All 400 registered validation-only measurements are complete. Each fresh control used exactly four BERT-base ranks; active measurements were exact-provenance reuse of the frozen A1 validation outputs. No checkpoint was retrained or mutated, and held-out test remained locked.

| Family | Control | Mean balanced | 95% t interval | Change | Preservation | Target |
|---|---|---:|---:|---:|---:|---:|
| ccrgb | t2b_editor_active | 0.299126 | [0.269683, 0.328569] | 0.284615 | 0.313636 | 0.308333 |
| ccrgb | t2b_editor_zeroed | 0.299126 | [0.269683, 0.328569] | 0.284615 | 0.313636 | 0.308333 |
| ccrgb | no_instruction_baseline | 0.299126 | [0.269683, 0.328569] | 0.284615 | 0.313636 | 0.308333 |
| ccrgb | prompting_baseline | 0.294580 | [0.271202, 0.317958] | 0.284615 | 0.304545 | 0.308333 |
| cladder | t2b_editor_active | 0.520516 | [0.514734, 0.526298] | 0.533333 | 0.507698 | 0.505082 |
| cladder | t2b_editor_zeroed | 0.520516 | [0.514734, 0.526298] | 0.533333 | 0.507698 | 0.505082 |
| cladder | no_instruction_baseline | 0.520516 | [0.514734, 0.526298] | 0.533333 | 0.507698 | 0.505082 |
| cladder | prompting_baseline | 0.517776 | [0.512156, 0.523395] | 0.537417 | 0.498135 | 0.509432 |
| com2 | t2b_editor_active | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.000000 |
| com2 | t2b_editor_zeroed | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.000000 |
| com2 | no_instruction_baseline | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.000000 |
| com2 | prompting_baseline | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.000000 |
| wiqa | t2b_editor_active | 0.500000 | [0.500000, 0.500000] | 0.000000 | 1.000000 | 0.000000 |
| wiqa | t2b_editor_zeroed | 0.500000 | [0.500000, 0.500000] | 0.000000 | 1.000000 | 0.000000 |
| wiqa | no_instruction_baseline | 0.500000 | [0.500000, 0.500000] | 0.000000 | 1.000000 | 0.000000 |
| wiqa | prompting_baseline | 0.500000 | [0.500000, 0.500000] | 0.000000 | 1.000000 | 0.000000 |
| xor | t2b_editor_active | 0.504630 | [0.501549, 0.507711] | 0.425792 | 0.583468 | 0.501188 |
| xor | t2b_editor_zeroed | 0.496858 | [0.494937, 0.498778] | 0.410562 | 0.583153 | 0.410687 |
| xor | no_instruction_baseline | 0.496858 | [0.494937, 0.498778] | 0.410562 | 0.583153 | 0.410687 |
| xor | prompting_baseline | 0.499015 | [0.497244, 0.500787] | 0.414340 | 0.583690 | 0.434292 |

## Paired comparisons

### ccrgb

- `primary_masking` (t2b_editor_zeroed minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `secondary_editor_contribution` (t2b_editor_active minus t2b_editor_zeroed): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `prompting_vs_no_instruction` (prompting_baseline minus no_instruction_baseline): -0.004545; 95% t [-0.011429, +0.002338]; paired graph bootstrap [-0.004167, -0.002143].

### cladder

- `primary_masking` (t2b_editor_zeroed minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `secondary_editor_contribution` (t2b_editor_active minus t2b_editor_zeroed): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `prompting_vs_no_instruction` (prompting_baseline minus no_instruction_baseline): -0.002740; 95% t [-0.006836, +0.001356]; paired graph bootstrap [-0.006553, +0.004178].

### com2

- `primary_masking` (t2b_editor_zeroed minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `secondary_editor_contribution` (t2b_editor_active minus t2b_editor_zeroed): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `prompting_vs_no_instruction` (prompting_baseline minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].

### wiqa

- `primary_masking` (t2b_editor_zeroed minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `secondary_editor_contribution` (t2b_editor_active minus t2b_editor_zeroed): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `prompting_vs_no_instruction` (prompting_baseline minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].

### xor

- `primary_masking` (t2b_editor_zeroed minus no_instruction_baseline): +0.000000; 95% t [+0.000000, +0.000000]; paired graph bootstrap [+0.000000, +0.000000].
- `secondary_editor_contribution` (t2b_editor_active minus t2b_editor_zeroed): +0.007772; 95% t [+0.004763, +0.010781]; paired graph bootstrap [+0.006237, +0.009366].
- `prompting_vs_no_instruction` (prompting_baseline minus no_instruction_baseline): +0.002157; 95% t [+0.001320, +0.002995]; paired graph bootstrap [+0.001326, +0.002902].

Prompting is reported separately from the primary masking check and secondary editor-contribution comparison. All results are validation-only; no held-out test record was opened.
