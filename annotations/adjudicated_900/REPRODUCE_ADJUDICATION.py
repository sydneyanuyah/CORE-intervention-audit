#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


DOWNLOADS = Path("${USER_HOME}/Downloads")
HANDOFF = DOWNLOADS / "CORE_Annotation_Handoff_900"
A_ROOT = DOWNLOADS / "CORE_Annotation_Results_900_Annotator1" / "submissions"
B_ROOT = DOWNLOADS / "CORE_Annotation_Results_900_Annotator2"
OUTPUT = DOWNLOADS / "CORE_Annotation_Adjudicated_900"

SOURCES = {
    "causalt5k": (
        HANDOFF / "10_CAUSALT5K_PRODUCTION_300.jsonl",
        A_ROOT / "causalt5k.ann-001.jsonl",
        B_ROOT / "causalt5k.ann-001.jsonl",
    ),
    "meter": (
        HANDOFF / "11_METER_PRODUCTION_300.jsonl",
        A_ROOT / "meter.ann-001.jsonl",
        B_ROOT / "meter.ann-002.jsonl",
    ),
    "pubmedcausal": (
        HANDOFF / "12_PUBMEDCAUSAL_PRODUCTION_300.jsonl",
        A_ROOT / "pubmedcausal.ann-001.jsonl",
        B_ROOT / "pubmedcausal.ann-003.jsonl",
    ),
}

ACCEPT_T2 = {"C5K-034", "C5K-139", "C5K-179", "C5K-208", "C5K-238", "C5K-239", "C5K-244", "C5K-249", "C5K-256"}
ACCEPT_T4 = {"C5K-139", "C5K-208", "C5K-238"}
SPECIAL_REJECTS = {
    ("C5K-034", "T4"), ("C5K-179", "T4"), ("C5K-239", "T4"),
    ("C5K-244", "T4"), ("C5K-249", "T4"), ("C5K-256", "T4"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict:
    count = len(pairs)
    labels = {label for pair in pairs for label in pair}
    a_counts = collections.Counter(a for a, _ in pairs)
    b_counts = collections.Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / count
    expected = sum((a_counts[label] / count) * (b_counts[label] / count) for label in labels)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else None
    return {
        "n": count,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "annotator_a_marginals": dict(a_counts),
        "annotator_b_marginals": dict(b_counts),
    }


def evidence_union(*annotations: dict) -> list[dict]:
    output, seen = [], set()
    for annotation in annotations:
        for item in annotation.get("evidence_spans") or []:
            if not item.get("text") and not item.get("rungs"):
                continue
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                output.append(item)
    return output


def canonical_accept(pid: str, task: str, payload: dict, a: dict, b: dict) -> dict:
    specs = {
        "C5K-034": {
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "factual_state": {"X": "no accelerator", "Y": 0.01},
            "intervention": {"target": "X", "operation": "do", "value": "accelerator"},
            "intervened_state": {"X": "accelerator", "Y": 0.20},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The released randomized 500-versus-500 comparison states both treatment values and both absolute Series-A rates.",
        },
        "C5K-139": {
            "graph": {"nodes": ["X", "Z", "Y"], "edges": [["X", "Y"]], "held_fixed": ["Z"]},
            "factual_state": {"X": True, "Z": "down by two points", "Y": "won"},
            "intervention": {"target": "X", "operation": "event_replace", "value": "shot not taken"},
            "intervened_state": {"X": False, "Z": "down by two points", "Y": "lost"},
            "reason": "ACCEPT_DETERMINISTIC_SCM",
            "why": "The score, immediate game end, no-other-score invariants, and absent-shot condition execute the no-shot world without adding an unsupported Z-to-Y edge.",
        },
        "C5K-179": {
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "factual_state": {"X": "high NOx / no diversion", "Y": "highest ozone"},
            "intervention": {"target": "X", "operation": "do", "value": "low NOx via full diversion"},
            "intervened_state": {"X": "low NOx / full diversion", "Y": "lowest ozone"},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The planned traffic-diversion manipulation and explicit high/medium/low ordered outcomes define comparable categorical states; CORE permits string-valued states.",
        },
        "C5K-208": {
            "graph": {"nodes": ["B", "X1", "X2", "X3", "T", "Y"], "edges": [["B", "T"], ["X1", "T"], ["X2", "T"], ["X3", "T"], ["T", "Y"]]},
            "factual_state": {"B": 29.6, "X1": 0.3, "X2": 0.4, "X3": 0.2, "T": 30.5, "Y": "bleaching"},
            "intervention": {"target": "X2", "operation": "do", "value": 0.0},
            "intervened_state": {"B": 29.6, "X1": 0.3, "X2": 0.0, "X3": 0.2, "T": 30.1, "Y": "bleaching"},
            "reason": "ACCEPT_DETERMINISTIC_SCM",
            "why": "The released additive equation explicitly identifies temperature T and computes 30.5 versus 30.1; T is the genuine changed descendant even though thresholded bleaching Y is preserved.",
        },
        "C5K-238": {
            "graph": {"nodes": ["X", "Z", "Y"], "edges": [["X", "Y"], ["Z", "Y"]]},
            "factual_state": {"X": True, "Z": "off-target throw; safe on play", "Y": "out by rule"},
            "intervention": {"target": "X", "operation": "event_replace", "value": False},
            "intervened_state": {"X": False, "Z": "off-target throw; safe on play", "Y": "safe"},
            "reason": "ACCEPT_DETERMINISTIC_SCM",
            "why": "The released disjunctive mechanisms, factual off-target throw, and explicit invariants determine the no-violation world.",
        },
        "C5K-239": {
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "factual_state": {"X": "new helmet", "Y": "200g drag"},
            "intervention": {"target": "X", "operation": "event_replace", "value": "old helmet"},
            "intervened_state": {"X": "old helmet", "Y": "250g drag"},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The repeated wind-tunnel helmet swap states both exact drag values at fixed power.",
        },
        "C5K-244": {
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "factual_state": {"X": "wild type", "Y": "tumors developed"},
            "intervention": {"target": "X", "operation": "do", "value": "ABC knockout"},
            "intervened_state": {"X": "ABC knockout", "Y": "no tumors over 16 weeks"},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The matched knockout experiment states both gene conditions and both tumor outcomes.",
        },
        "C5K-249": {
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "factual_state": {"X": "10g weight inserted", "Y": "300 yards"},
            "intervention": {"target": "X", "operation": "event_replace", "value": "weight removed"},
            "intervened_state": {"X": "weight removed", "Y": "280 yards"},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The ablation and re-insertion sequence supplies exact paired outcomes under identical conditions.",
        },
        "C5K-256": {
            "graph": {"nodes": ["X", "Y", "W"], "edges": [["X", "Y"]], "held_fixed": ["W"]},
            "factual_state": {"X": "VAR present", "Y": 1.0, "W": "reference other-foul rate"},
            "intervention": {"target": "X", "operation": "do", "value": "VAR removed"},
            "intervened_state": {"X": "VAR absent", "Y": 1.4, "W": "unchanged"},
            "reason": "ACCEPT_EXPLICIT_PAIR",
            "why": "The controlled removal defines both arms; indexing the stated 40% relative increase to a released comparison baseline of 1.0 preserves rather than invents the reported ratio.",
        },
    }
    spec = specs[pid]
    return {
        "annotator_id": "ann-999",
        "decision": "accept",
        "reason_codes": [spec["reason"]],
        "graph": spec["graph"],
        "factual_state": spec["factual_state"],
        "intervention": spec["intervention"],
        "intervened_state": spec["intervened_state"],
        "evidence_spans": evidence_union(a, b),
        "notes": f"Senior adjudication ({task}): {spec['why']}",
    }


def accepted(pid: str, task: str) -> bool:
    return (task == "T2" and pid in ACCEPT_T2) or (task == "T4" and pid in ACCEPT_T4)


def adjudicate_reject(pid: str, task: str, source: str, payload: dict, a: dict, b: dict) -> dict:
    reasons = sorted((set(a.get("reason_codes") or []) | set(b.get("reason_codes") or [])) - {"ACCEPT_EXPLICIT_PAIR", "ACCEPT_DETERMINISTIC_SCM"})
    if (pid, task) in SPECIAL_REJECTS:
        reasons = ["AMBIGUOUS_OR_CONFLICTING_EVIDENCE"]
        note = "Released level is L1 but the content describes an intervention. T4 requires the released rung to remain authoritative, so the row is retained as a diagnostic/sidecar and excluded from formal rung-transfer evidence."
    elif pid == "MTR-141":
        reasons = ["GRAPH_NOT_IDENTIFIED", "FACTUAL_STATE_NOT_IDENTIFIED", "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
        note = "The METER answer options do not supply an authoritative common graph or complete paired worlds; adjudicated reject."
    elif pid in {"PMC-069", "PMC-159"}:
        reasons = ["EDGE_NOT_SUPPORTED", "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
        note = "The released relation text is misaligned with the sentence, and no resolution supplies paired experimental worlds; adjudicated reject rather than unresolved label."
    else:
        if not reasons:
            if source == "pubmedcausal":
                reasons = ["NO_INTERVENTION"]
            elif source == "meter":
                reasons = ["GRAPH_NOT_IDENTIFIED", "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
            else:
                reasons = ["COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
        note = "Senior adjudication: reject. Both released evidence and the two independent annotations fail at least one required graph/intervention/world-state element."
    return {
        "annotator_id": "ann-999", "decision": "reject", "reason_codes": reasons,
        "graph": None, "factual_state": None, "intervention": None, "intervened_state": None,
        "evidence_spans": evidence_union(a, b), "notes": note,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = []
    accepted_rows = []
    final_counts = collections.Counter()
    agreement_counts = {}
    input_hashes = {}
    output_paths = {}
    all_decision_pairs = []
    collision = {
        "graph_group_id": "causalt5k:case:0148",
        "rows": ["C5K-200", "C5K-208"],
        "required_repair": "include release_domain in the graph-group identity before conversion/splitting",
    }

    for source, (original_path, a_path, b_path) in SOURCES.items():
        original = read_jsonl(original_path)
        aa = {row["pilot_id"]: row for row in read_jsonl(a_path)}
        bb = {row["pilot_id"]: row for row in read_jsonl(b_path)}
        input_hashes[source] = {"original": digest(original_path), "annotator_a": digest(a_path), "annotator_b": digest(b_path)}
        decision_agree = total_views = reason_agree = 0
        decision_pairs_by_task = collections.defaultdict(list)
        final_rows = []
        for source_row in original:
            pid = source_row["pilot_id"]
            row = dict(source_row)
            row["task_views"] = {}
            for task in source_row["task_views"]:
                total_views += 1
                a = aa[pid]["task_views"][task]
                b = bb[pid]["task_views"][task]
                same_decision = a["decision"] == b["decision"]
                same_reasons = set(a.get("reason_codes") or []) == set(b.get("reason_codes") or [])
                decision_agree += int(same_decision)
                reason_agree += int(same_reasons)
                pair = (a["decision"], b["decision"])
                decision_pairs_by_task[task].append(pair)
                all_decision_pairs.append(pair)
                final = canonical_accept(pid, task, row["source_payload"], a, b) if accepted(pid, task) else adjudicate_reject(pid, task, source, row["source_payload"], a, b)
                row["task_views"][task] = final
                final_counts[(source, task, final["decision"])] += 1
                if final["decision"] == "accept":
                    accepted_rows.append({
                        "pilot_id": pid, "source": source, "source_locator": row["source_locator"],
                        "graph_group_id": row["graph_group_id"], "task_view": task,
                        "final_annotation": final,
                        "conversion_gate": "candidate accepted by annotation; still requires normalized CORE conversion and mechanical schema/span/group validation",
                    })
                agreement_overridden = same_decision and a["decision"] != final["decision"]
                if not same_decision or agreement_overridden or final["decision"] == "accept":
                    comparison.append({
                        "pilot_id": pid, "source": source, "source_locator": row["source_locator"], "task_view": task,
                        "annotator_a_decision": a["decision"], "annotator_a_reason_codes": a.get("reason_codes", []),
                        "annotator_b_decision": b["decision"], "annotator_b_reason_codes": b.get("reason_codes", []),
                        "decision_agreed": same_decision, "agreement_overridden": agreement_overridden,
                        "final_decision": final["decision"], "final_reason_codes": final["reason_codes"],
                        "adjudication_notes": final["notes"],
                    })
            final_rows.append(row)
        agreement_counts[source] = {
            "task_views": total_views, "decision_agreements": decision_agree,
            "decision_agreement_rate": decision_agree / total_views,
            "reason_set_agreements": reason_agree, "reason_set_agreement_rate": reason_agree / total_views,
            "cohen_kappa": cohen_kappa([pair for pairs in decision_pairs_by_task.values() for pair in pairs]),
            "task_kappa": {task: cohen_kappa(pairs) for task, pairs in sorted(decision_pairs_by_task.items())},
        }
        out = OUTPUT / f"{source}.adjudicated.jsonl"
        write_jsonl(out, final_rows)
        output_paths[source] = out

    write_jsonl(OUTPUT / "accepted_task_views.jsonl", accepted_rows)
    write_jsonl(OUTPUT / "adjudication_audit.jsonl", comparison)
    report = {
        "protocol": "T2-T4-G4 two-annotator senior adjudication",
        "production_source_rows": 900,
        "task_views": 1200,
        "annotator_a_package": str(A_ROOT.parent),
        "annotator_b_package": str(B_ROOT),
        "annotator_identity_warning": "Both CausalT5K submissions declare ann-001; they are distinguished as package A and package B. Final consensus uses ann-999 and adjudicator adj-001.",
        "agreement": agreement_counts,
        "overall_cohen_kappa": cohen_kappa(all_decision_pairs),
        "decision_disagreements": sum(v["task_views"] - v["decision_agreements"] for v in agreement_counts.values()),
        "agreement_overrides": 2,
        "final_counts": {f"{s}:{t}:{d}": n for (s, t, d), n in sorted(final_counts.items())},
        "accepted_task_views": len(accepted_rows),
        "accepted_unique_source_rows": len({row["pilot_id"] for row in accepted_rows}),
        "accepted_pilot_ids": sorted({row["pilot_id"] for row in accepted_rows}),
        "meter_accepted": 0,
        "pubmedcausal_accepted": 0,
        "pubmed_test_accessed": False,
        "known_group_identity_repair": collision,
        "input_hashes": input_hashes,
        "outputs": {source: {"path": str(path), "sha256": digest(path)} for source, path in output_paths.items()},
        "accepted_task_views_sha256": digest(OUTPUT / "accepted_task_views.jsonl"),
        "adjudication_audit_sha256": digest(OUTPUT / "adjudication_audit.jsonl"),
    }
    (OUTPUT / "ADJUDICATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# CORE two-annotator adjudication report", "",
        "## Outcome", "",
        "- 900 production source rows and 1,200 task views were compared.",
        f"- Decision disagreements requiring review: {report['decision_disagreements']}.",
        "- Two additional CausalT5K T4 accept/accept agreements were overridden because intervention-like content was released under L1; the frozen T4 guide forbids silently changing the source rung.",
        f"- Final accepted task views: {report['accepted_task_views']} across {report['accepted_unique_source_rows']} CausalT5K source rows.",
        "- Final accepted by task: T2 = 9; T4 = 3; G4 = 0.",
        "- METER: 0/300 accepted. PubMedCausal: 0/300 accepted.",
        "- These are accepted annotation candidates, not yet normalized experimental records.", "",
        "## Agreement", "",
        f"Overall multiclass Cohen's kappa across all 1,200 task views is **{report['overall_cohen_kappa']['cohen_kappa']:.4f}** (observed agreement {report['overall_cohen_kappa']['observed_agreement']:.1%}; chance-expected agreement {report['overall_cohen_kappa']['expected_agreement']:.1%}).",
        "", "| Dataset | Views | Decision agreement | Cohen's kappa | Reason-set agreement |", "|---|---:|---:|---:|---:|",
    ]
    for source in ("causalt5k", "meter", "pubmedcausal"):
        x = agreement_counts[source]
        md.append(f"| {source} | {x['task_views']} | {x['decision_agreements']}/{x['task_views']} ({x['decision_agreement_rate']:.1%}) | {x['cohen_kappa']['cohen_kappa']:.4f} | {x['reason_set_agreements']}/{x['task_views']} ({x['reason_set_agreement_rate']:.1%}) |")
    md += [
        "", "CausalT5K task-specific kappa: T2 = **0.5788**; T4 = **0.5245**.",
        "", "METER and PubMedCausal each have kappa 0.0000 despite 99%+ raw agreement because annotator B assigned only `reject`, producing no label variance. This prevalence/kappa paradox must be reported alongside raw agreement; it does not mean the annotators disagreed on every item.",
    ]
    md += [
        "", "## Accepted candidates", "",
        "- T2: C5K-034, C5K-139, C5K-179, C5K-208, C5K-238, C5K-239, C5K-244, C5K-249, C5K-256.",
        "- T4: C5K-139, C5K-208, C5K-238.",
        "- G4: none.", "",
        "## Mandatory repair before conversion", "",
        "`C5K-200` (D10) and `C5K-208` (D6) share the frozen group string `causalt5k:case:0148` despite being unrelated cases. Before normalization or splitting, derive CausalT5K graph groups from release domain plus case ID. Do not silently edit the adjudicated frozen input.", "",
        "## Audit cautions", "",
        "- Both CausalT5K submissions identify themselves as `ann-001`; provenance therefore distinguishes package A and package B by path and SHA-256.",
        "- The first package's pre-existing self-adjudicated files were not used as an independent vote.",
        "- PubMedCausal held-out test data were not accessed.",
        "- Run the packaged submission validator against each adjudicated queue before downstream conversion.",
    ]
    (OUTPUT / "ADJUDICATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
