#!/usr/bin/env python3
"""Create an immutable corrected L3 manifest from an existing expansion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    prefix = source["cells"][0]["cell_id"].split(":", 1)[0]
    methods = ("lora_matched", "loreft", "o2")
    cells = []
    for cell in source["cells"]:
        if cell["method"] not in methods:
            continue
        updated = dict(cell)
        updated["cell_id"] = f"{prefix}-independent-v2:{cell['method']}:{cell['seed']}"
        updated["output"] = f"outputs/{prefix}-independent-v2/{cell['method']}/seed-{cell['seed']}"
        cells.append(updated)
    result = {
        "protocol": f"{source['protocol']}_independent_methods_v2",
        "supersedes_invalid_protocol": source["protocol"],
        "backbone_frozen": True,
        "comparison_scope": "frozen_feature_operator_probe",
        "methods": list(methods), "seeds": source["seeds"], "cells": cells,
        "source_manifest": source["source_manifest"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_amendment": source["source_amendment"],
        "source_amendment_sha256": source["source_amendment_sha256"],
        "training_steps": source["training_steps"], "batch_size": 256,
        "evaluation_split": "validation", "test_evaluated": False,
        "operator_contract": {
            "lora_matched": "parameter_matched_frozen_feature_adapter_v1",
            "loreft": "command_conditioned_loreft_style_residual_v1",
            "o2": "state_independent_command_indexed_o2_v1",
        },
        "reporting_caveat": "lora_matched is a frozen-feature parameter-matched adapter, not Q/V LoRA on backbone weights",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.output), "cells": len(cells)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
