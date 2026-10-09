# T2, T4, and G4 Structured Annotation Guide

Version: 1.1
Purpose: create auditable candidate graph/intervention pairs for T2, T4, and G4 without inventing causal structure or counterfactual outcomes.

## Frozen 300-per-dataset pilot

The annotation labor pilot is now frozen as three independent 300-item queues, not one 300-item pool:

- CausalT5K: 300 production items, balanced at 10 items per D1-D10 x L1-L3 stratum. Each row has separate T2 and T4 annotation fields.
- METER: 300 production items formed from 100 intact context groups, with one discovery, intervention, and counterfactual question per group.
- PubMedCausal: 300 production relation candidates: 120 implicit/intra, 120 explicit/intra, 30 implicit/inter, and 30 explicit/inter. Only train and validation are used; test is not accessed.
- Five additional fully worked senior-annotator examples per dataset are excluded from the production queues. This makes 900 production items plus 15 teaching items.

The ordinary three-digit selection seed is `317`. The reproducible builder, queue hashes, and completed real-data examples are in the sibling data repository:

- Annotation Bible: `${USER_HOME}/Documents/core-data/docs/ANNOTATION_BIBLE_T2_T4_G4.md`
- Manifest: `${USER_HOME}/Documents/core-data/reports/annotation_pilot_300_manifest.json`
- Local generated queues: `${USER_HOME}/Documents/core-data/data/annotation_pilots/`

The queues remain third-party-derived local data and are intentionally not committed. The script, hashes, and instructions are committed. None of these candidates becomes formal experimental evidence until double annotation, adjudication, conversion, and all Section 8 gates pass.

## 1. Non-negotiable rule

Annotators may transcribe, normalize, and link information that is explicit in an authoritative source. They may not guess an edge, structural equation, factual value, intervention value, counterfactual value, or affected variable.

If the source does not determine a field, record the candidate in the annotation ledger as `reject` or `needs_adjudication`. Do not place it in the normalized CORE JSONL file. JSON `null` is allowed only where the CORE schema permits it; it is not a substitute for the factual/intervened state evidence required by these experiments.

Annotation is therefore a filtering and structuring task, not permission to manufacture a new gold standard. Creating genuinely new counterfactual labels requires a separately approved data-creation protocol, qualified domain experts, independent replication, and explicit provenance.

## 2. Files produced by annotation

Keep annotation workflow metadata outside the immutable CORE record because the schema rejects unexpected keys.

1. `annotations/<source>.ledger.jsonl`: one row for every candidate, including rejected candidates.
2. `annotations/<source>.adjudication.jsonl`: disagreements and final decisions.
3. `data/records/<source>.jsonl`: accepted CORE records only.
4. `data/records/<source>.rejected.jsonl`: rejected candidate plus exact reason.
5. `data/splits/<source>.train.txt`, `.validation.txt`, and `.test.txt`: record IDs only.
6. `reports/<source>_annotation_summary.json`: counts, annotator agreement, hashes, and rejection reasons.

Suggested ledger row:

```json
{
  "candidate_id": "causalt5k:T3-BucketD-0102",
  "source_locator": "D10/L2/T3-BucketD-0102",
  "annotator_id": "ann-003",
  "decision": "reject",
  "reason_code": "COUNTERFACTUAL_STATE_NOT_IDENTIFIED",
  "evidence_spans": [],
  "graph_group_id": "causalt5k:case:0102",
  "release_domain": "D10",
  "notes": "The prose names possible relations but does not identify both world states."
}
```

Use anonymous annotator IDs. Do not put annotator identity, domain labels, or adjudication notes into the normalized record.

## 3. Required CORE record

Every accepted row must have exactly these root keys:

```json
{
  "id": "source-stable-record-id",
  "source": "source-name",
  "source_id": "original-source-id",
  "structure_kind": "dag",
  "graph": {
    "nodes": ["X", "Y"],
    "edges": [["X", "Y"]],
    "node_text": {
      "X": ["released surface form for X"],
      "Y": ["released surface form for Y"]
    }
  },
  "chain": null,
  "factual": {
    "passage": "Source-grounded model input containing the intervention target text.",
    "state": {"X": 0, "Y": 0},
    "question": "Source-grounded factual question or null.",
    "answer": 0
  },
  "intervention": {
    "target": "X",
    "target_text": "released surface form for X",
    "target_span": [42, 69],
    "value": 1,
    "value_token": 1,
    "replacement_span": null,
    "kind": "value_set",
    "formal": "do(X = 1)",
    "text": "Source-grounded intervention instruction."
  },
  "intervened": {
    "passage": null,
    "state": {"X": 1, "Y": 1},
    "answer": 1
  },
  "descendants": ["Y"],
  "non_descendants": [],
  "probes": [
    {
      "variable": "X",
      "question": null,
      "answer_before": 0,
      "answer_after": 1,
      "required": "may change",
      "actually_changed": true
    },
    {
      "variable": "Y",
      "question": null,
      "answer_before": 0,
      "answer_after": 1,
      "required": "may change",
      "actually_changed": true
    }
  ],
  "provenance": {
    "fetched_utc": "YYYY-MM-DDTHH:MM:SSZ",
    "url": "exact-source-url",
    "file": "data/raw/source/file",
    "sha256": "64-lowercase-hex-characters",
    "converter": "src/convert_source_annotated.py",
    "converter_version": 1
  }
}
```

The numeric offsets above are illustrative; production spans must be calculated against the exact final passage and pass the schema validator.

### 3.1 Graph requirements

- `nodes` are stable structural IDs, not arbitrary display strings.
- Every edge must be explicitly supported by the source or an authoritative attached graph.
- The graph must be acyclic and every edge endpoint must occur in `nodes`.
- `node_text`, when present, maps every node exactly once to one or more released surface forms.
- If the source provides only an ordered event path rather than a DAG, use `structure_kind: "chain"`, set `graph` to `null`, and preserve at least two ordered events in `chain`.
- Do not convert correlation, temporal order, or an answer option into an edge unless the source explicitly defines it causally.

### 3.2 World-state requirements

- `factual.state` and `intervened.state` must use the same variable IDs and comparable scalar values.
- Values may be strings, numbers, booleans, or `null`, but must use a documented closed vocabulary within an annotation batch.
- The intervention must be surgical: replace the mechanism/value of the target and propagate only through supported descendants.
- Both world states must be stated by the source, mechanically derived from an explicit deterministic SCM, or verified independently by two experts under a separately approved data-creation protocol.
- A multiple-choice answer alone is not a complete world state.

### 3.3 Intervention addressing

- `target` is the stable graph ID; `target_text` is the exact surface text in `factual.passage`.
- `target_span` is a zero-based, half-open `[start, end]` character span and must slice exactly to `target_text`.
- `value_set` requires a non-null `value_token` and a null `replacement_span`.
- `event_replace` requires the old and replacement event mentions to appear in the factual passage, a null `value_token`, and an exact `replacement_span` pointing to the replacement text.
- `formal` must be a faithful normalization such as `do(X = 0)`; it cannot add information absent from the source.

### 3.4 Descendants and probes

- Exclude the intervention target from both `descendants` and `non_descendants`.
- The two lists must be disjoint and reference only declared graph nodes.
- A `must not change` probe must have equal before/after answers and `actually_changed: false`.
- An accepted record needs at least one `may change` probe with a genuine observed difference.
- A structurally valid intervention with no observed effect belongs in the rejection file with reason `NO_OBSERVED_MAY_CHANGE_EFFECT`; retain it for sensitivity analyses only if a later protocol explicitly permits it.

## 4. Annotation workflow

1. **Freeze the source.** Record revision, raw-file hash, licence, and exact locator before annotation.
2. **Identify the leakage group.** Use the complete scenario, graph, paper, patient, experiment, or paragraph—not the individual question—as `graph_group_id`.
3. **Extract variables.** Assign stable IDs and copy exact released groundings into evidence spans.
4. **Extract structure.** Record only supported edges or a supported chain. Cite the exact text/table/figure location for every edge.
5. **Record the factual world.** List every variable value required for the probes.
6. **Record the intervention.** Identify target, operation, value, textual grounding, and exact character span.
7. **Record the intervened world.** Use explicit paired evidence or deterministic SCM execution. Never infer it from plausibility.
8. **Create probes.** Include target, changed descendants, and supported invariants.
9. **Independent review.** A second annotator repeats steps 3–8 without seeing the first labels.
10. **Adjudicate.** Accept only exact agreements or documented expert adjudications. Unresolved disagreements are rejected.
11. **Validate mechanically.** Run the CORE schema validator, graph acyclicity check, span check, observed-effect check, and leakage audit.
12. **Freeze splits and hashes.** Split by `graph_group_id`; never split questions from one scenario or paper across train/validation/test.

Recommended decision codes:

- `ACCEPT_EXPLICIT_PAIR`
- `ACCEPT_DETERMINISTIC_SCM`
- `NO_INTERVENTION`
- `GRAPH_NOT_IDENTIFIED`
- `EDGE_NOT_SUPPORTED`
- `FACTUAL_STATE_NOT_IDENTIFIED`
- `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`
- `TARGET_NOT_GROUNDED`
- `NO_OBSERVED_MAY_CHANGE_EFFECT`
- `AMBIGUOUS_OR_CONFLICTING_EVIDENCE`
- `LICENCE_NOT_CLEARED`
- `DUPLICATE_OR_LEAKAGE_RISK`

## 5. T2 — CausalT5K held-out-domain annotation

### 5.1 T2-specific objective

T2 needs O3 training and validation examples grouped by the ten released domains. The domain label is evaluation metadata, not a causal variable. Preserve it in the annotation ledger and split sidecar. Leave one complete domain out per registered fold; never distribute one case or graph across folds.

CausalT5K L1 association questions normally have no intervention and should be rejected. L2/L3 questions may be accepted only when the released item or an attached authoritative source supplies a complete graph and paired world states. A prose `causal_structure` hint is evidence to review, not automatically a complete graph.

### 5.2 T2 annotation fields outside CORE JSON

```json
{
  "release_domain": "D1",
  "release_level": "L2",
  "case_id": "2.001",
  "graph_group_id": "causalt5k:2.001",
  "edge_evidence": [{"edge": ["Z", "Y"], "source_span": [120, 176]}],
  "state_evidence": [{"world": "factual", "source_locator": "item/table/row"}],
  "fold_role": "assigned only after annotation freeze"
}
```

### 5.3 Six T2 worked examples

These are annotation patterns. “Accept” applies only when the described evidence actually exists in the frozen source.

#### T2 example 1 — Explicit binary SCM and paired state: accept

Source evidence: an item explicitly states `Z -> X`, `Z -> Y`, `X -> Y`, gives factual values `Z=1, X=1, Y=1`, and states that under `do(X=0)` the outcome remains `Y=1`.

- Decision: `ACCEPT_EXPLICIT_PAIR`.
- Graph: nodes `[Z, X, Y]`; edges `[[Z, X], [Z, Y], [X, Y]]`.
- Factual state: `{Z: 1, X: 1, Y: 1}`.
- Intervention: `do(X=0)` with target span grounded to the exact exposure phrase.
- Intervened state: `{Z: 1, X: 0, Y: 1}`.
- Descendants: `[Y]`; non-descendants: `[Z]`.
- Probes: X changes; Y may change but does not; Z must not change.
- Leakage group: the complete case, not this question.

#### T2 example 2 — Deterministic equations permit execution: accept

Source evidence: `X := Z`, `Y := X XOR Z`, factual exogenous value `Z=1`, and the requested intervention `do(X=0)`.

- Decision: `ACCEPT_DETERMINISTIC_SCM`.
- Execute the released equations: factual `{Z:1, X:1, Y:0}`; intervened `{Z:1, X:0, Y:1}`.
- Record the equations and execution trace in the ledger; the normalized record contains graph, states, intervention, and probes.
- Graph: `Z -> X`, `Z -> Y`, `X -> Y`.
- Observed effect exists at X and Y, so the record satisfies the primary contract.
- A reviewer must reproduce both states independently.

#### T2 example 3 — L1 causal claim with no intervention: reject

Source pattern: “Winter birth is associated with schizophrenia; does seasonal infection cause the higher rate?” with a yes/no label and possible confounders.

- Decision: `NO_INTERVENTION`.
- Reason: neither a surgical target/value nor a paired intervened world is supplied.
- Do not reinterpret the claim “infection causes schizophrenia” as `do(infection=1)`.
- Preserve the variables, domain, label, rationale, and source locator in the rejection ledger.

#### T2 example 4 — Prose arrows but missing states: reject

Source pattern: `Application deterrence -> stricter rules` and `Application deterrence -> fraud incidence`, plus a narrative that reported fraud fell.

- Decision: `FACTUAL_STATE_NOT_IDENTIFIED` and `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`.
- The arrows may support a partial graph, but “fell sharply” is not a complete variable state and no paired world under a named intervention is identified.
- Do not fill omitted values from the answer rationale.

#### T2 example 5 — Multiple plausible graphs: adjudicate, then usually reject

Source pattern: magnesium use occurs after a high-stress blood-pressure reading; a later calm reading is lower. The prose permits `stress -> measurement`, `measurement -> supplement`, and possibly `supplement -> blood pressure`, but does not identify which structural model generated the case.

- Initial decision: `AMBIGUOUS_OR_CONFLICTING_EVIDENCE`.
- Two annotators independently propose edges and mark evidence spans.
- If the authoritative item does not uniquely resolve the graph and both world states, final decision is reject.
- The observed before/after measurements must not be relabelled as factual/intervened outcomes because context changed simultaneously.

#### T2 example 6 — Explicit intervention but no observed effect: reject from primary set

Source evidence: graph and factual state are complete; under `do(X=0)` all recorded variables, including X because it was already 0, remain unchanged.

- Decision: `NO_OBSERVED_MAY_CHANGE_EFFECT`.
- The graph may be valid, but the primary CORE schema requires at least one genuine may-change difference.
- Store the fully structured candidate in the rejection ledger for an optional no-effect sensitivity protocol.
- Do not change the intervention value merely to force a positive example.

## 6. T4 — METER/CausalT5K rung-transfer annotation

### 6.1 T4-specific objective

T4 compares changing-only, imagining-only, and joint learning across Pearl-style rungs. Annotation must preserve the rung supplied by the source:

- `discovery`: observational causal-discovery question;
- `intervention`: explicit action or `do` operation;
- `counterfactual`: alternative world conditioned on a stated factual world.

Store `rung`, `context_id`, `question_index`, and `graph_group_id` in a sidecar. The normalized CORE training set still requires an intervention pair. Discovery-only rows may support reader pretraining only under a later frozen protocol; they are not intervention records.

The three T4 methods are model treatments, not labels annotators should improvise. Annotation supplies aligned rung evidence. A separate immutable manifest must define which fields each treatment sees.

### 6.2 Six T4 worked examples

#### T4 example 1 — Complete aligned three-rung scenario: accept intervention and counterfactual records

Source evidence supplies one explicit graph `A -> B -> C`, factual state `{A:1,B:1,C:1}`, intervention `do(B=0)`, resulting state `{A:1,B:0,C:0}`, and a counterfactual question whose answer is C=0.

- Group discovery, intervention, and counterfactual questions under one `graph_group_id`.
- Discovery row remains sidecar/pretraining evidence unless separately enabled.
- Create an accepted intervention record with B as target and C as changed descendant.
- Create the counterfactual-aligned record only if it uses the same explicit factual world and SCM.
- Split the entire scenario as one unit.

#### T4 example 2 — Intervention multiple-choice question without graph: reject

Source pattern: “What happens if a manufacturer includes a guide with a standard sundial?” followed by plausible answer options.

- Decision: `GRAPH_NOT_IDENTIFIED` and `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`.
- The correct option does not establish a complete DAG or values for alternative outcomes.
- Do not make every noun a node or every option an intervened state.

#### T4 example 3 — Counterfactual narrative with two simultaneous changes: reject or split only with explicit SCM

Source pattern: “What would differ if the hurricane were slower and brought heavy rain instead of powerful winds?”

- This changes speed, rainfall, and wind together; it is not a single surgical intervention.
- Decision: `AMBIGUOUS_OR_CONFLICTING_EVIDENCE` unless the source explicitly defines an `event_replace` operation and its downstream state.
- Do not arbitrarily choose “storm speed” as the sole target.

#### T4 example 4 — Explicit event replacement in a released chain: accept

Source evidence gives the chain `powerful winds -> roof damage -> displacement`, factual event “powerful winds,” replacement event “light winds,” and explicitly states the resulting chain has no roof damage or displacement.

- Structure: `chain`; `graph: null`.
- Intervention kind: `event_replace`.
- The factual passage must contain both old and replacement mentions so exact target and replacement spans can be recorded.
- Before/after probes cover roof damage and displacement.
- Accept only if the alternative outcomes are explicitly released, not merely intuitive.

#### T4 example 5 — Causal-discovery question only: retain in ledger, reject as CORE intervention record

Source pattern: “Why did Gaara join the search?” with one correct explanatory option.

- Decision: `NO_INTERVENTION` for the normalized CORE intervention file.
- Preserve the discovery label and scenario group in the T4 sidecar.
- It may later be used by `changing_only` only if the model-treatment protocol explicitly permits observational reader data.
- It cannot supply the imagining or joint outcome by itself.

#### T4 example 6 — Complete graph but leakage across rungs: reject split and repair grouping

Source supplies valid discovery, intervention, and counterfactual rows from the same context, but an initial split assigns discovery to train and counterfactual to validation.

- Record-level labels may be correct, but split decision is `DUPLICATE_OR_LEAKAGE_RISK`.
- Move the complete context/graph group to exactly one split before freeze.
- Recompute all split hashes and counts.
- Never preserve a source question-level split when it leaks the same causal scenario across partitions.

## 7. G4 — PubMedCausal intervention annotation

### 7.1 G4-specific objective

G4 compares baseline and O3 on implicit/intra causal language. The existing PubMedCausal row supplies a sentence and cause/effect spans, not paired interventions. A sentence can become an accepted CORE record only when an authoritative linked experimental source explicitly identifies both treatment/control conditions and their outcomes.

For the currently pinned sentence-only release:

- a cause span is not automatically an intervention target;
- an effect span is not automatically an intervened value;
- “causal association” is not a paired world state;
- `Implicit` and `Intra` remain text-relation metadata in the sidecar;
- source-provided train/validation/test partitions must be preserved, and the existing test partition must not be opened during development.

If linked abstracts, tables, or full text are added, that is a new source acquisition. Freeze its licence, URL, revision, and hash before annotation.

### 7.2 Six G4 worked examples

#### G4 example 1 — Randomized treatment and control outcomes explicitly reported: accept

Authoritative evidence: a randomized study reports mean systolic blood pressure of 142 in control and 134 under a named drug, with otherwise matched conditions.

- Graph: `drug_assignment -> systolic_bp`; include additional nodes only when explicitly modeled.
- Factual/control state: `{drug_assignment:0, systolic_bp:142}`.
- Intervention state: `{drug_assignment:1, systolic_bp:134}`.
- Intervention: `do(drug_assignment=1)` grounded to the exact treatment phrase.
- Preserve population, endpoint, timepoint, and units in sidecar evidence; do not mix outcomes from different timepoints.
- Group all rows from the same trial arm comparison together.

#### G4 example 2 — Gene knockout with measured wild-type and knockout states: accept

Authoritative evidence explicitly reports protein expression in wild type and after a specific knockout.

- Graph: `gene_activity -> protein_expression` only if the experiment/source supports that direction.
- Factual state uses wild type; intervened state uses knockout.
- Intervention value must reflect the released coding, e.g. `0`/`inactive`, consistently across the batch.
- Accept when both measured outcomes and experimental conditions are explicit.

#### G4 example 3 — Mendelian-randomization “causal association”: reject

Source pattern: “Higher insulin levels are causally associated with increased endometrial-cancer risk.”

- Decision: `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`.
- The spans identify a relation but provide neither a surgical insulin intervention nor paired outcome states.
- Do not convert “higher” and “increased risk” into arbitrary binary values.
- Retain the relation for a possible redefined span-extraction G4, not current CORE G4.

#### G4 example 4 — Observational association or mechanistic speculation: reject

Source pattern: “Adiponectin increases over time and is associated with disability and mortality; it may be a marker or causal factor.”

- Decision: `EDGE_NOT_SUPPORTED` plus `NO_INTERVENTION`.
- Association and speculation do not identify an intervention graph.
- Multiple plausible directions/confounders prevent a unique causal record.

#### G4 example 5 — Experimental effect direction but missing comparable values: reject

Source pattern: “Treatment significantly reduced inflammation,” without the control value, treatment value, endpoint definition, or timepoint.

- Decision: `FACTUAL_STATE_NOT_IDENTIFIED` and `COUNTERFACTUAL_STATE_NOT_IDENTIFIED`.
- Statistical significance alone is not a state value.
- Annotators may retrieve a frozen authoritative table only if the acquisition protocol permits it and provenance is recorded; they may not estimate values from prose.

#### G4 example 6 — Conflicting abstract and table: adjudicate

The abstract says the treatment increased outcome Y, while the frozen results table shows a decrease at the registered endpoint.

- Initial decision: `AMBIGUOUS_OR_CONFLICTING_EVIDENCE`.
- Two domain reviewers verify arm coding, endpoint, direction, and timepoint.
- Adjudication may accept the table-backed pair only with a documented correction/interpretation and exact evidence locator.
- If conflict remains unresolved, reject. Do not choose the label that makes the causal sentence easier to model.

## 8. Quality-control and acceptance thresholds

For a batch to be eligible for preregistration:

- 100% of accepted records pass `src/schema.py` validation.
- 100% of target and replacement spans match exact source text.
- 100% of edges and before/after values have evidence locators or deterministic execution traces.
- 100% of records have exactly one leakage-group split assignment.
- 100% of accepted records contain an observed may-change effect.
- Two independent annotations exist for every accepted record.
- Graph, target, intervention value, factual state, intervened state, and probe agreement are reported separately.
- All disagreements are adjudicated; unresolved cases are rejected.
- No development decision uses the held-out test partition.
- Raw files, ledgers, accepted/rejected outputs, split manifests, converters, and summaries are SHA-256 frozen.

Suggested minimum agreement target before scaling annotation is 0.90 exact agreement on accept/reject and 0.85 exact agreement on graph edges and state values. These are workflow quality thresholds, not permission to accept unresolved examples.

## 9. Final annotator checklist

Before marking a candidate `accept`, answer yes to every question:

1. Is the intervention explicit and surgical?
2. Is its target grounded by an exact passage span?
3. Is every graph edge supported by cited evidence?
4. Are factual and intervened values explicitly supported or deterministically executable?
5. Is at least one may-change probe genuinely different?
6. Are invariants labelled `must not change` and unchanged?
7. Are descendants/non-descendants valid and disjoint?
8. Is the full scenario/paper/graph assigned to one leakage group?
9. Did an independent annotator reproduce the record?
10. Does the final normalized JSON pass the exact CORE schema?

If any answer is no, do not place the row in the accepted dataset.
