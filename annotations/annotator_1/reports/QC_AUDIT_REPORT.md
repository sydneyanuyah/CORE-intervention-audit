# Quality-control report: ann-001 pass, senior audit, adjudication

## What was done

Nine hundred production items were annotated by one annotator identity, `ann-001`,
across 1,200 task views (CausalT5K carries an independent T2 and T4 form on each row).
Every decision was reached by reading the released item against the evidence contract in
`ANNOTATOR_BRIEF.md`; no decision was assigned by script, pattern or heuristic. Assembly of
the hand-written annotations into the frozen queue files, character-offset arithmetic, and
the mechanical checks below were performed programmatically.

Three independent senior reviewers then audited the pass. Each was given every non-reject
annotation in its dataset plus a random sample of rejects (seed 317), and asked to form its
own judgement first and challenge what it found. Their objections were put to a fourth
reviewer, `adj-001`, who ruled on each from the released evidence rather than deferring to
either party.

## Mechanical checks — all clean

- `40_VALIDATE_SUBMISSION.py` passes on all three ann-001 files and all three adjudicated files.
- 437 PubMedCausal character-offset spans re-sliced against their released sentence: 437 exact,
  0 mismatches. 192 released cause/effect strings recorded with `span: null` were confirmed
  genuinely absent from their sentence (case changes, tense and number changes, nominalisations,
  dropped conjuncts, source typos such as `blindess`, `aggrevating`, `challanges`).
- 1,978 evidence quotes across CausalT5K and METER checked as verbatim substrings of their own
  item's payload: 0 failures.
- 1,200 notes, 1,200 distinct. No boilerplate; maximum pairwise similarity within an audited
  sample was 0.19.
- Immutable fields (`pilot_id`, `source`, `source_locator`, `graph_group_id`, `source_payload`)
  byte-identical to the frozen queues.
- No PubMedCausal test row was opened, sampled or referenced.

## What the auditors found

**CausalT5K (66 annotations audited).** All 46 audited rejects were upheld — the auditor found
no row that in fact supplied all four elements. The errors were concentrated entirely in the
non-reject half: three accepts were challenged as manufacturing a missing element, and all
seven `needs_adjudication` calls were challenged, five as under-specified rows that should have
been plain rejects and two as accepts withheld over a metadata question the same annotator had
already answered on other rows.

**METER (23 annotations audited).** Zero wrong decisions, zero unjustified reason codes. The
auditor specifically confirmed the discipline the contract is meant to enforce: the annotator
twice declined `FACTUAL_STATE_NOT_IDENTIFIED` on rows where the factual world is genuinely
stated, and withheld `TARGET_NOT_GROUNDED` wherever the target had exact surface text. Four
hygiene defects were noted (an under-applied code, two rejects carrying non-null structures,
one mis-targeted evidence span).

**PubMedCausal (24 annotations audited).** Zero wrong decisions. Every offset sliced exactly and
every null span was confirmed as a genuine non-occurrence — the part of the work most easily
faked held up against the source. Two evidence records were found where the `text` entered was
itself verbatim but had been recorded with a null span; one reason code was struck and several
were normalised for cross-item consistency.

## Adjudication outcome

`adj-001` ruled on 44 contested annotations. Fourteen decisions were overturned:

| Item | ann-001 | Final | Ground |
|---|---|---|---|
| C5K-208 T2, T4 | accept | reject | The may-change difference existed only on a node the annotator introduced; the declared outcome Y is unchanged at 30.5 °C and 30.1 °C. `NO_OBSERVED_MAY_CHANGE_EFFECT`. |
| C5K-256 T2 | accept | reject | The factual "index 1.00" and intervened "1.40 x" values are annotator-supplied; only a relative 40% increase is released. |
| C5K-256 T4 | needs_adjudication | reject | Fails independently on the absent absolute penalty rates. |
| C5K-217 T2, T4 | needs_adjudication | reject | Only an "8% higher" gap is released, with no turnout level for either band. A missing world is a rejection. |
| C5K-248 T2 | needs_adjudication | reject | Every branch of the escalated question ends in a rejection. |
| C5K-168 T2 | needs_adjudication | reject | The mediator/common-cause conflict is real but not decisive; no 5%-commission world is released under either reading. |
| C5K-034 T4, C5K-239 T4 | needs_adjudication | accept | All four elements are on the row; the L1 level string carries none of the weight, exactly as in the annotator's own C5K-244 and C5K-249 accepts. |
| MTR-141 T4 | needs_adjudication | reject | Both parties wrong: a JTWC forecast is a prediction about the actual world, and "in part because" defeats surgicality. |
| PMC-069 G4, PMC-159 G4 | needs_adjudication | reject | Whether the relation was mis-keyed cannot be settled from released text, and the label is determinate either way. The row-alignment defect is escalated upstream instead. |

The remaining 30 rulings upheld the decision and corrected reason codes, evidence spans or
structured-field hygiene: `ACCEPT_DETERMINISTIC_SCM` corrected to `ACCEPT_EXPLICIT_PAIR` on
C5K-238 where no equations are released; an unsupported edge struck from C5K-139 T4; degenerate
null and empty-string evidence entries repaired on C5K-201 and C5K-010; the PMC-228 and PMC-222
spans given their real offsets; `EDGE_NOT_SUPPORTED` dropped from PMC-183 and PMC-036;
`NO_INTERVENTION` withdrawn from rows naming a real manipulation; bidirectionality coding
normalised; and the duplicate flag extended to PMC-246, which shares a sentence with PMC-063 and
PMC-071 across batch boundaries.

## Result

After adjudication: 12 accepts, 1,188 rejects, 0 unresolved adjudications across 1,200 task views.
All 12 accepts are CausalT5K rows carrying an explicitly released paired world.

## Standing limitation

This is a single annotator pass. Section 8 of the full guide requires two independent annotations
for every accepted record before it becomes experimental evidence, and the audit above is not a
substitute — the auditors saw the first annotations. The 12 accepted rows in particular should be
re-annotated blind by a second annotator before conversion, and the auditors' own reasoning should
not be shown to that annotator.

## Data defects worth referring to the queue owner

- `graph_group_id` collisions between unrelated cases: `causalt5k:case:4.21` (C5K-005 / C5K-006),
  `causalt5k:case:5.329` (C5K-053 / C5K-054), `causalt5k:case:0148` (C5K-200 D10 sociology /
  C5K-208 D6 climate). The group id appears to be built from the bare case number, which collides
  across buckets, and for T2 this would tie rows from different held-out domains into one group.
- PubMedCausal row-alignment defects: `train/10086/relation/12` (PMC-069) and
  `train/10481/relation/14` (PMC-159) carry cause/effect strings topically unrelated to their
  released sentence.
- One PubMedCausal sentence, `validation/154`, appears three times in the queue as PMC-063,
  PMC-071 and PMC-246.
- Released cause/effect strings frequently do not occur verbatim in their sentence (192 of 629
  span records), including truncations (`eosinophil coun`, `causalit`), misspellings and
  mojibake. Character offsets cannot be recovered for these without a corrected release.
- Assorted CausalT5K rows carry internal conflicts between the scenario's `(X)`/`(Y)` tags and the
  `variables` block, empty `causal_structure`/`claim` fields, byte-identical conditional answers,
  and `case_id`/`source_id` mismatches. These are named item by item in the ledger notes.
- A recurring level/content mismatch: a number of rows describing randomised or ablation designs
  are released at L1. This does not change any decision here but bears directly on T4 rung transfer.
