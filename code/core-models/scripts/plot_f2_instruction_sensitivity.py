#!/usr/bin/env python3
"""Plot the audited F2 clean-score versus command-sensitivity mechanism result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repo / "reports/F2_INSTRUCTION_SENSITIVITY.json")
    parser.add_argument("--output", type=Path, default=repo / "reports/figures/F2_COMMAND_SENSITIVITY.svg")
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if report.get("completed_cells") != 40 or report.get("test_evaluated") is not False:
        raise ValueError("F2 sensitivity figure requires the complete validation-only report")

    width, height = 900, 620
    left, right, top, bottom = 105, 45, 90, 95
    x_min, x_max, y_min, y_max = 0.68, 0.93, 0.59, 0.79

    def xy(method: str) -> tuple[float, float]:
        row = report["methods"][method]
        x = row["changed_pair_vectors"] / row["pair_measurements"]
        y = row["mean_clean_two_edit_balanced"]
        px = left + (x - x_min) / (x_max - x_min) * (width - left - right)
        py = top + (y_max - y) / (y_max - y_min) * (height - top - bottom)
        return px, py

    points = {method: xy(method) for method in ("o2", "o3")}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append('<text x="450" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">State gating attenuates command following on CLadder</text>')
    for value in (0.60, 0.65, 0.70, 0.75):
        py = top + (y_max - value) / (y_max - y_min) * (height - top - bottom)
        svg.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#d9d9d9"/>')
        svg.append(f'<text x="{left-14}" y="{py+5:.1f}" text-anchor="end" font-family="Arial" font-size="14">{value:.2f}</text>')
    for value in (0.70, 0.75, 0.80, 0.85, 0.90):
        px = left + (value - x_min) / (x_max - x_min) * (width - left - right)
        svg.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#eeeeee"/>')
        svg.append(f'<text x="{px:.1f}" y="{height-bottom+26}" text-anchor="middle" font-family="Arial" font-size="14">{value:.0%}</text>')
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333" stroke-width="2"/>')
    x3, y3 = points["o3"]
    x2, y2 = points["o2"]
    svg.append(f'<line x1="{x3:.1f}" y1="{y3:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#777" stroke-width="2" stroke-dasharray="8 6"/>')
    for method, color in (("o2", "#1f77b4"), ("o3", "#d62728")):
        px, py = points[method]
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="{color}" stroke="white" stroke-width="3"/>')
        svg.append(f'<text x="{px+17:.1f}" y="{py-14:.1f}" font-family="Arial" font-size="20" font-weight="700">{method.upper()}</text>')
    svg.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-38}" text-anchor="middle" font-family="Arial" font-size="16">Pair predictions changed after command-value inversion</text>')
    svg.append(f'<text transform="translate(26 {(top+height-bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="16">Clean two-edit balanced score</text>')
    svg.append(f'<text x="{left}" y="{height-10}" font-family="Arial" font-size="12" fill="#555">Validation only · 20 checkpoints per method · 12,300 pairs per method</text>')
    svg.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
