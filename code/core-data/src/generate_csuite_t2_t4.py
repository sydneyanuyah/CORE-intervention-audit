"""Generate fresh T2/T4 development pairs from the pinned official CSuite simulator.

This script never opens released test.csv, interventions.json, or
counterfactuals.json.  It imports the official simulator at its pinned Causica
revision, captures each released SEM configuration, and draws new paired
interventional worlds with ordinary three-digit experiment seeds.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


CAUSICA_REVISION = "not-published"
PROTOCOL_T2 = "t2_csuite_generated_development_v1"
PROTOCOL_T4 = "t4_csuite_rung_views_v1"


def capture_configs(root: Path):
    sys.path.insert(0, str(root))
    import numpy as np
    import numpyro.distributions as dist
    from causica.data_generation.csuite import simulate

    configs = {}
    original = simulate.simulate_data

    def capture(*args, **kwargs):
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        configs[current[0]] = dict(bound.arguments)

    current = [""]
    builders = {
        "lingauss": lambda: simulate.two_node_lin(1, 1, "/private/tmp", dist.Normal(0, 1), dist.Normal(0, 1), "lingauss"),
        "linexp": lambda: simulate.two_node_lin(1, 1, "/private/tmp", dist.TransformedDistribution(dist.Exponential(1), dist.transforms.AffineTransform(loc=-1.0, scale=1.0)), dist.TransformedDistribution(dist.Exponential(1), dist.transforms.AffineTransform(loc=-1.0, scale=1.0)), "linexp"),
        "nonlingauss": lambda: simulate.two_node_nonlinear_gauss(1, 1, "/private/tmp"),
        "nonlin_simpson": lambda: simulate.nonlin_simpson(1, "/private/tmp"),
        "symprod_simpson": lambda: simulate.symprod_simpson(1, "/private/tmp"),
        "large_backdoor": lambda: simulate.large_backdoor(1, "/private/tmp"),
        "weak_arrows": lambda: simulate.weak_arrows(1, "/private/tmp"),
        "cat_to_cts": lambda: simulate.cat_to_cts(1, 1, "/private/tmp"),
        "cts_to_cat": lambda: simulate.cts_to_cat(1, 1, "/private/tmp"),
        "mixed_simpson": lambda: simulate.mixed_simpson(1, 1, "/private/tmp"),
        "large_backdoor_binary_t": lambda: simulate.large_backdoor(1, "/private/tmp", binary_treatment=True),
        "weak_arrows_binary_t": lambda: simulate.weak_arrows(1, "/private/tmp", binary_treatment=True),
        "mixed_confounding": lambda: simulate.mixed_confounding(1, 1, "/private/tmp"),
        "cat_chain": lambda: simulate.cat_chain(1, 1, "/private/tmp"),
        "cat_collider": lambda: simulate.cat_collider(1, 1, "/private/tmp"),
    }
    simulate.simulate_data = capture
    try:
        for name, builder in builders.items():
            current[0] = name; builder()
    finally:
        simulate.simulate_data = original
    if set(configs) != set(builders):
        raise ValueError("failed to capture every CSuite SEM")
    return configs


def draw(model, intervention: dict, count: int, seed_value: int):
    import jax.random as jr
    import numpy as np
    from numpyro.handlers import do, seed, trace
    from causica.data_generation.csuite.pyro_utils import expand_model

    expanded = seed(expand_model(do(model, data=intervention), count, "core_dev_plate"), jr.PRNGKey(seed_value))
    traced = trace(expanded).get_trace()
    result = {}
    for index in range(100):
        name = f"x{index}"
        if name not in traced: break
        values = np.asarray(traced[name]["value"])
        if values.ndim == 1: values = values[:, None]
        result[name] = values.reshape(count, -1)
    for name, value in intervention.items():
        raw = np.asarray(value).reshape(1, -1)
        result[name] = np.repeat(raw, count, axis=0)
    return result


def state_at(samples: dict, row: int) -> dict[str, float | int | list]:
    import numpy as np
    state = {}
    for name in sorted(samples, key=lambda value: int(value[1:])):
        values = samples[name][row]
        clean = [int(v) if np.issubdtype(values.dtype, np.integer) else round(float(v), 8) for v in values]
        state[name] = clean[0] if len(clean) == 1 else clean
    return state


def clean_value(value):
    import numpy as np
    values = np.asarray(value).reshape(-1)
    clean = [int(v) if np.issubdtype(values.dtype, np.integer) else round(float(v), 8) for v in values]
    return clean[0] if len(clean) == 1 else clean


def graph_text(adjacency) -> str:
    edges = [f"x{i} -> x{j}" for i in range(len(adjacency)) for j in range(len(adjacency)) if int(adjacency[i, j])]
    return "Causal graph: " + (", ".join(edges) if edges else "no directed edges") + "."


def state_text(state: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in state.items())


def write_jsonl(path: Path, rows) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True); digest = hashlib.sha256(); count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line); digest.update(line.encode()); count += 1
    return count, digest.hexdigest()


def generate(causica_root: Path, t2_output: Path, t4_output: Path, summary_path: Path, samples_per_seed: int) -> dict:
    revision = subprocess.check_output(["git", "-C", str(causica_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != CAUSICA_REVISION: raise ValueError(f"Causica revision mismatch: {revision}")
    configs = capture_configs(causica_root)
    family_counts = collections.Counter()

    def base_pairs(seeds):
        for family, cfg in sorted(configs.items()):
            model, adjacency = cfg["numpyro_model"], cfg["adjacency_matrix"]
            idx = int(cfg["intervention_idx"]); name = f"x{idx}"
            primary_value, reference_value = cfg["intervention_value"], cfg["reference_value"]
            for seed_value in seeds:
                # Same random key pairs exogenous noise; only the surgical write differs.
                primary = draw(model, {name: primary_value}, samples_per_seed, seed_value * 2)
                reference = draw(model, {name: reference_value}, samples_per_seed, seed_value * 2)
                for row_index in range(samples_per_seed):
                    family_counts[family] += 1
                    yield family, seed_value, row_index, adjacency, idx, primary_value, reference_value, state_at(reference, row_index), state_at(primary, row_index)

    def t2_rows():
        for family, seed_value, row_index, adjacency, idx, primary_value, reference_value, factual, changed in base_pairs(range(501, 506)):
            yield {
                "id": f"t2:{family}:{seed_value}:{row_index:04d}", "protocol": PROTOCOL_T2,
                "sem_family": family, "generation_seed": seed_value, "split": "development_generated",
                "graph_group_id": f"csuite:{family}", "world_group_id": f"csuite:{family}:{seed_value}:{row_index:04d}",
                "graph_text": graph_text(adjacency), "factual_state": factual,
                "intervention": {"target": f"x{idx}", "reference_value": clean_value(reference_value), "value": clean_value(primary_value)},
                "intervened_state": changed,
                "rendered_input": f"{graph_text(adjacency)} Factual world: {state_text(factual)}. Apply do(x{idx}={clean_value(primary_value)}).",
                "rendered_target": state_text(changed), "test_evaluated": False,
            }

    def t4_rows():
        for family, seed_value, row_index, adjacency, idx, primary_value, reference_value, factual, changed in base_pairs(range(511, 516)):
            graph = graph_text(adjacency); change = f"Change x{idx} from {clean_value(reference_value)} to {clean_value(primary_value)}."; imagine = f"Imagine the alternative world resulting from do(x{idx}={clean_value(primary_value)}) given factual state {state_text(factual)}."
            shared = {"sem_family": family, "generation_seed": seed_value, "split": "development_generated", "graph_group_id": f"csuite:{family}", "world_group_id": f"csuite:{family}:{seed_value}:{row_index:04d}", "factual_state": factual, "intervened_state": changed, "rendered_target": state_text(changed), "test_evaluated": False}
            for arm, text in (("changing_only", f"{graph} {change}"), ("imagining_only", f"{graph} {imagine}"), ("joint", f"{graph} {change} {imagine}")):
                yield {"id": f"t4:{family}:{seed_value}:{row_index:04d}:{arm}", "protocol": PROTOCOL_T4, "arm": arm, "rendered_input": text, **shared}

    t2_count, t2_hash = write_jsonl(t2_output, t2_rows())
    t4_count, t4_hash = write_jsonl(t4_output, t4_rows())
    summary = {
        "causica_revision": revision, "sem_families": sorted(configs), "sem_family_count": len(configs),
        "samples_per_seed": samples_per_seed, "t2_seeds": list(range(501, 506)), "t4_seeds": list(range(511, 516)),
        "t2_rows": t2_count, "t2_sha256": t2_hash, "t4_rows": t4_count, "t4_sha256": t4_hash,
        "paired_noise": True, "released_test_members_opened": False, "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causica-root", type=Path, required=True)
    parser.add_argument("--t2-output", type=Path, default=Path("data/t2/csuite_pairs.jsonl"))
    parser.add_argument("--t4-output", type=Path, default=Path("data/t4/csuite_rung_views.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/csuite_t2_t4_generation_summary.json"))
    parser.add_argument("--samples-per-seed", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(generate(args.causica_root, args.t2_output, args.t4_output, args.summary, args.samples_per_seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
