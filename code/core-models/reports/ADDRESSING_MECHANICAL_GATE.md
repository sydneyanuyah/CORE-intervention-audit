# Addressing mechanical gate

Date: 2026-09-04

These are bounded engineering measurements on CCR.GB validation, not scientific A1 results. Every run used BERT-base on exactly four NVIDIA A16 GPUs, authoritative graph/world sidecars, and exact non-padding validation. No held-out test record was loaded.

## A2 frozen-checkpoint path

Both conditions loaded checkpoint SHA-256 `not-published` and performed no training.

| Condition | Records | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| T2-b editor active | 12 | 0.4167 | 0.1471 |
| T2-b editor zeroed | 12 | 0.4167 | 0.1471 |

The equal values are expected from a one-epoch, 96-record smoke whose editor remains near initialization. This establishes frozen checkpoint reuse, not edit attribution. The later preregistered 400-cell A2 measurement supersedes this bounded smoke; see `reports/A2_MEASUREMENT.md`.

## A3 frozen-checkpoint path

All three conditions loaded checkpoint SHA-256 `not-published` and performed no training.

| Span | Records | Accuracy | Pointer mass | Pointer top-1 |
|---|---:|---:|---:|---:|
| Correct | 12 | 0.4167 | 0.2385 | 0.0833 |
| Adjacent | 12 | 0.4167 | 0.2403 | 0.0833 |
| Random | 12 | 0.4167 | 0.2504 | 0.3333 |

The required random-span collapse did not occur. This is unsurprising for the bounded one-epoch smoke, but it means the A3 scientific gate is not passed and A1 must not start. The comparison does establish that correct, adjacent, and random overrides execute concurrently against one immutable validation-selected checkpoint and report pointer diagnostics with complete coverage.

## Gate decision

- Production DDP, sidecar grouping, frozen measurement, and exact validation: pass.
- A2 attribution result: complete in `reports/A2_MEASUREMENT.md`; the primary masking check is exactly zero in every family, while editor contribution is positive only on XOR and zero on the four real-family validation sets.
- A3 random-span collapse: fail on the bounded smoke.
- A1 evidence launch: closed.

## Full three-source development checkpoint and repeated A3

The full T3-b BERT-base development run used CLadder, WIQA, and CCR.GB on four
GPUs for three epochs with a per-GPU batch size of 32, held-out structured
paraphrases, and explicit per-variable supervision. Exact validation covered
4,706 records in 486 graph groups. The validation-selected checkpoint SHA-256
was `not-published`.

All three frozen measurements loaded those identical checkpoint bytes and set
`trained_during_measurement: false`.

| Span | Records | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Correct | 4,706 | 0.4934 | 0.3335 | 0.0934 | 0.0907 | 0.8314 |
| Adjacent | 4,706 | 0.4934 | 0.3335 | 0.0933 | 0.0886 | 0.8315 |
| Random | 4,706 | 0.4934 | 0.3335 | 0.0924 | 0.0886 | 0.8310 |

The full development checkpoint also fails A3. Random-span addressing causes
zero task-accuracy or macro-F1 collapse, while correct-span pointer top-1 is
only 9.07%. The near-uniform pointer diagnostics and span-invariant predictions
show that the selected reader is not using the addressed T3 edit reliably.
This is a model-learning failure, not a coverage or execution failure: pointer
coverage and unseen-paraphrase coverage were both 100%.

The scientific boundary remains closed. Do not start A1 evidence runs from this
checkpoint. The next development action is to make pointer supervision explicit
in the training objective, then require correct-span pointer learning and
random-span task collapse on validation before reopening A1.

## Pointer-supervised development checkpoint

Adding correct-slot cross-entropy increased correct-span pointer top-1 from
9.07% to 41.22% and correct-slot mass from 9.34% to 35.22%. Accuracy also moved
from 49.34% to 49.83% and macro-F1 from 0.3335 to 0.3881. Frozen adjacent and
random controls reduced pointer top-1 to 34.23% and 30.15%, respectively, so
the pointer now responds to the address.

Task accuracy nevertheless remained exactly 49.83% under correct, adjacent,
and random spans. Pointer supervision therefore fixes address learning but not
the causal dependence of the answer on the edit. Inspection found that upper
decoder layers still allowed post-edit slot and question queries to reread the
original passage. A strict post-edit attention bottleneck now blocks that path;
the next full development checkpoint tests pointer supervision plus this
bottleneck. A1 remains closed.

## Post-edit bottleneck result

The bottleneck checkpoint selected epoch 2 by validation macro-F1. All frozen
conditions used checkpoint SHA-256
`not-published`
and performed no training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|
| Correct | 0.4996 | 0.3326 | 0.3456 | 0.3833 | 0.7952 |
| Adjacent | 0.4996 | 0.3326 | 0.3181 | 0.3315 | 0.7952 |
| Random | 0.4996 | 0.3326 | 0.2696 | 0.3081 | 0.7952 |

The bottleneck is mechanically valid but does not produce task collapse. Both
task and per-variable accuracy are span-invariant. The remaining shortcut is
objective-level: per-variable supervision is dominated by unchanged variables,
so the model can minimize it while ignoring the intervention-induced target and
downstream changes. Further training with the same objective is not warranted.
The next development objective must separately supervise the intervened target
and changed downstream variables, with preservation retained as its own term.
A1 remains closed.

## Explicit transition result

The shared-head before/after transition checkpoint selected epoch 3. All three
frozen controls used checkpoint SHA-256
`not-published`
and performed no training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|
| Correct | 0.5011 | 0.3497 | 0.3501 | 0.4042 | 0.7318 |
| Adjacent | 0.5013 | 0.3500 | 0.3231 | 0.3381 | 0.7323 |
| Random | 0.5011 | 0.3497 | 0.2672 | 0.2960 | 0.7329 |

Explicit transition supervision improved state semantics but did not bind the
independent pooled-question task head to those states. A3 still fails and A1
remains closed. An audit found exactly one question-bearing probe, with exact
task-label agreement, for all 12 CCR.GB and 4,547 WIQA validation records. The
next gate uses that authoritative queried slot as the causal task readout and
retains the existing decoder only for the 147 uncovered CLadder records.

## Causal task readout result

The queried-slot checkpoint selected epoch 3. Frozen correct, adjacent, and
random controls used checkpoint SHA-256
`not-published`.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|
| Correct | 0.5368 | 0.4613 | 0.3444 | 0.3976 | 0.7295 |
| Adjacent | 0.5361 | 0.4609 | 0.3236 | 0.3625 | 0.7307 |
| Random | 0.5355 | 0.4597 | 0.2711 | 0.2998 | 0.7319 |

The causal readout raised task performance but random addressing reduced
accuracy by only 0.13 percentage points, which is not collapse. The remaining
mechanical shortcut is the soft pointer's edit mixture over every slot. A3 and
A1 remain closed; the next development gate uses a straight-through hard
one-slot pointer rather than another objective-weight search.

## Hard pointer result

The straight-through hard-pointer checkpoint selected epoch 3. All controls
used checkpoint SHA-256
`not-published`.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|
| Correct | 0.5295 | 0.5015 | 0.3470 | 0.3950 | 0.7287 |
| Adjacent | 0.5295 | 0.5016 | 0.3255 | 0.3508 | 0.7287 |
| Random | 0.5297 | 0.5015 | 0.2758 | 0.3034 | 0.7286 |

A3 still fails. Code tracing shows that after-state and queried-task heads read
the immediate editor output rather than slots contextualized by the post-edit
upper layers. Consequently, a discrete edit to an upstream target cannot
change the downstream queried-slot readout. The next gate places `answer_after`
and task heads after post-edit propagation while keeping `answer_before` on the
pre-edit slots.

## Post-edit propagation result

The propagated-slot checkpoint selected epoch 3, and all frozen controls used
checkpoint SHA-256
`not-published`.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Variable accuracy |
|---|---:|---:|---:|---:|---:|
| Correct | 0.5584 | 0.5208 | 0.3536 | 0.3955 | 0.8393 |
| Adjacent | 0.5589 | 0.5211 | 0.3254 | 0.3491 | 0.8387 |
| Random | 0.5574 | 0.5202 | 0.2714 | 0.2949 | 0.8386 |

Propagation improved state quality, but A3 still fails. Every one of the 4,547
WIQA validation passages contains a fixed `[TARGET]` marker visible to the
shared trunk in all three arms. The next gate masks this marker from trunk
attention, eliminating that address bypass while retaining the span-selected
instruction. A1 remains closed.

## Structural-support propagation result

The target-plus-descendant structural-support checkpoint selected epoch 3.
All frozen controls performed no training and reused checkpoint SHA-256
`not-published`.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.7862 | 0.6508 | 0.3516 | 0.4127 | 0.4780 | 0.7617 | 0.9272 |
| Adjacent | 0.7907 | 0.6531 | 0.3029 | 0.3304 | 0.4780 | 0.7665 | 0.9258 |
| Random | 0.7748 | 0.6448 | 0.2687 | 0.3056 | 0.4780 | 0.7498 | 0.9271 |

Structural support materially improved correct-span task accuracy, but random
spans reduced it by only 1.15 percentage points. Target accuracy was invariant,
change accuracy fell only 1.19 points, and preservation stayed invariant.
Therefore the registered A3 random-span collapse criterion is still not met;
A3 fails and A1 remains closed. These group metrics cover the explicit probe
labels available in validation (159 target, 4,720 change, and 12,727
preservation cells) and do not use test data.

## Slot/question isolation result

Blocking post-edit structural slots from reading question-token keys did not
change the A3 conclusion. All controls reused checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.7877 | 0.6210 | 0.3541 | 0.4067 | 0.4780 | 0.7517 | 0.8876 |
| Adjacent | 0.7888 | 0.6215 | 0.3103 | 0.3228 | 0.4780 | 0.7553 | 0.8825 |
| Random | 0.7890 | 0.6216 | 0.2799 | 0.3100 | 0.4780 | 0.7538 | 0.8853 |

Random-span task accuracy was 0.13 points higher than correct-span accuracy,
and target/change/preservation metrics remained effectively invariant. The
question-to-slot bypass is eliminated, but the unmasked target phrase remains
visible to the pre-edit trunk as ordinary passage text. The next architecture
gate must isolate that phrase from the trunk and admit it only through the
address/instruction channel. A3 and A1 remain closed.

## Target-span isolation result

The isolated-target checkpoint selected epoch 3. All frozen controls reused
checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.7433 | 0.5988 | 0.3127 | 0.3755 | 0.5597 | 0.6919 | 0.8931 |
| Adjacent | 0.7448 | 0.5998 | 0.2719 | 0.3213 | 0.5660 | 0.6964 | 0.8845 |
| Random | 0.7429 | 0.5986 | 0.2654 | 0.3147 | 0.5472 | 0.6932 | 0.8858 |

Removing the target phrase from the shared trunk reduced the bypass but did
not produce random-span collapse. A converter audit then found that normalized
WIQA retained only symbolic node IDs (`V`, `W`, and similar) even though the
released influence-graph file supplies natural-language groundings for every
node. Mapping an unseen phrase to an abstract slot was therefore mechanically
underdetermined. Converter version 3 now preserves those released groundings
as optional `graph.node_text`; no labels or test outcomes are inferred.

## Grounded-slot result

Grounded slots improved correct-span pointer top-1 from 37.55% to 53.63%, but
frozen correct/adjacent/random task accuracies remained
60.01%/59.99%/59.84%. All arms reused checkpoint SHA-256
`not-published`
without training. Correct-to-random pointer mass fell from 0.3838 to 0.2634,
yet task accuracy fell only 0.17 points, so A3 still fails.

The causal task adapter was truncating the shared variable head from five
classes to the four benchmark classes. That discarded `no_effect` precisely
when a wrong address left the queried variable unchanged, forcing a less/more
guess instead of a causal failure. The adapter now retains `no_effect` as an
abstention prediction that is counted wrong against the four-class task gold.
This changes no gold label and uses no test data.

## No-effect-aware readout result

The no-effect-aware checkpoint selected epoch 3. Frozen correct, adjacent, and
random controls reused checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.4915 | 0.3788 | 0.3520 | 0.5363 | 0.4780 | 0.4905 | 0.7229 |
| Adjacent | 0.4883 | 0.3788 | 0.2728 | 0.3217 | 0.4780 | 0.4873 | 0.6793 |
| Random | 0.4868 | 0.3787 | 0.2565 | 0.3139 | 0.4780 | 0.4858 | 0.6801 |

The abstention class makes wrong-span state degradation visible, but random
task accuracy falls only 0.47 points and target accuracy is unchanged. A3
therefore still fails. Architecture tracing found that the post-edit decoder
allows every structural slot to attend to every other slot after the explicit
target-plus-descendant support edit. A wrong edit can consequently spread back
into the queried slot outside the declared graph support. The next gate blocks
cross-slot mixing during post-edit decoding; question tokens can still read all
slots, and each slot retains its self state. A1 remains closed.

## Post-edit structural-slot isolation result

The isolated-slot checkpoint selected epoch 3. All frozen controls reused
checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.5848 | 0.4850 | 0.3496 | 0.5206 | 0.5220 | 0.5847 | 0.7507 |
| Adjacent | 0.5841 | 0.4856 | 0.2663 | 0.3126 | 0.5220 | 0.5841 | 0.7297 |
| Random | 0.5795 | 0.4854 | 0.2519 | 0.3156 | 0.5220 | 0.5792 | 0.7325 |

Cross-slot isolation reduced random-span task accuracy by only 0.53 points;
A3 still fails. The pointer itself retains 31.56% random-span top-1 accuracy
because its slot keys are lower-trunk contextual states that contain the whole
record, rather than the released slot-name grounding alone. The next gate uses
uncontextualized grounded slot-name embeddings for address selection while
retaining contextual slots for editing. This removes record-context target
priors from the pointer without changing supervision, coefficients, or data.
A1 remains closed.

## Grounded-name pointer-key result

The grounded-name-key checkpoint selected epoch 3. Frozen controls reused
checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.6572 | 0.5845 | 0.2519 | 0.3714 | 0.4717 | 0.6553 | 0.6815 |
| Adjacent | 0.6507 | 0.5815 | 0.2112 | 0.3064 | 0.4780 | 0.6492 | 0.6732 |
| Random | 0.6468 | 0.5795 | 0.1998 | 0.2988 | 0.4717 | 0.6453 | 0.6738 |

Removing whole-record context from pointer keys improved correct task quality,
but random accuracy fell only 1.04 points and A3 still fails. A direct lexical
audit shows that released WIQA target text identifies the authoritative slot
by maximal token overlap for 4,379/4,547 validation records (96.31%). The
learned pointer is not exploiting that grounding. The next fail-closed gate
restricts pointer candidates to maximally overlapping grounded slot names and
applies no edit when an override span has no slot grounding. This uses input
text only, not labels or test records. A1 remains closed.

## Fail-closed lexical-grounding result

The lexical-grounded checkpoint selected epoch 3. Frozen controls reused
checkpoint SHA-256
`not-published`
without training.

| Span | Accuracy | Macro-F1 | Pointer mass | Pointer top-1 | Target | Change | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 0.4981 | 0.3964 | 0.9602 | 0.9707 | 0.5220 | 0.4981 | 0.9814 |
| Adjacent | 0.1602 | 0.3221 | 0.1509 | 0.1912 | 0.5220 | 0.1612 | 0.9147 |
| Random | 0.3164 | 0.3667 | 0.1670 | 0.1702 | 0.5220 | 0.3169 | 0.8118 |

The registered A3 mechanism now passes. Random-span pointer top-1 collapses
from 97.07% to 17.02%, and task accuracy collapses by 18.17 percentage points
(36.47% relative). The adjacent control falls by 33.79 points. Preservation
degrades less than changed-variable performance, as expected for a selective
edit. Target-group accuracy is invariant because WIQA does not label its
intervention target as a probe and the remaining 159 target probes come from
CCR.GB/CLadder, where symbolic slot grounding does not change under these text
overrides. A3 is passed on validation; A1 architecture screening is open. No
held-out test records were accessed.
