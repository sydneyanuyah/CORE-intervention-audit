#!/usr/bin/env python3
"""Run repaired large-CCR.GB F2 cells in a clean output namespace."""

from __future__ import annotations

import argparse

import dispatch_f2_ccrgb as operators


ARTIFACT_SHA = "not-published"
PAIR_SHA = "not-published"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("operators", "baselines"), required=True)
    args = parser.parse_args()
    operators.ARTIFACT_SHA = ARTIFACT_SHA
    operators.PAIR_SHA = PAIR_SHA
    if args.phase == "operators":
        operators.Cell.output = property(
            lambda self: operators.ROOT / f"outputs/f2-repaired/ccrgb/{self.method}/seed-{self.seed}"
        )
        return operators.main()

    import dispatch_f2_ccrgb_baselines as baselines
    baselines.ARTIFACT_SHA = ARTIFACT_SHA
    baselines.PAIR_SHA = PAIR_SHA
    baselines.Cell.output = property(
        lambda self: baselines.ROOT / f"outputs/f2-repaired/ccrgb/{self.method}/seed-{self.seed}"
    )
    return baselines.main()


if __name__ == "__main__":
    raise SystemExit(main())
