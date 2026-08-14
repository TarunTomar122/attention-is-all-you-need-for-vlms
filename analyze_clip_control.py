"""Summarize the minimal one-seed CLIP-family backbone control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a4", type=Path, required=True)
    parser.add_argument("--s4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a4 = torch.load(args.a4, map_location="cpu", weights_only=False)
    s4 = torch.load(args.s4, map_location="cpu", weights_only=False)
    if a4["ids"] != s4["ids"]: raise ValueError("CLIP predictions are not paired")
    metrics = {}
    for name in ("acc_iou_0.5", "iou", "pointing", "target_mass"):
        metrics[name] = {"A4": float(a4["metrics"][name].float().mean()), "S4": float(s4["metrics"][name].float().mean())}
        metrics[name]["A4_minus_S4"] = metrics[name]["A4"] - metrics[name]["S4"]
    payload = {"protocol": {"backbone": "openai/clip-vit-large-patch14-336", "seeds": [0], "note": "minimal one-seed backbone control"}, "metrics": metrics, "n": len(a4["ids"])}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# CLIP-family backbone control", "", "This is a minimal one-seed matched RefCOCOg test because the control is intended to test backbone transfer, not add a new matrix.", "", "| Metric | A4 | S4 | A4−S4 |", "| --- | ---: | ---: | ---: |"]
    for name, row in metrics.items():
        scale = 100 if name in ("acc_iou_0.5", "pointing") else 1
        unit = "%" if scale == 100 else ""
        lines.append(f"| {name} | {scale*row['A4']:.4f}{unit} | {scale*row['S4']:.4f}{unit} | {scale*row['A4_minus_S4']:+.4f}{unit} |")
    lines += ["", f"N = {len(a4['ids'])}. This result is descriptive and should not be treated as a three-seed significance test.", ""]
    (args.output.parent / "clip_control_summary.md").write_text("\n".join(lines)); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
