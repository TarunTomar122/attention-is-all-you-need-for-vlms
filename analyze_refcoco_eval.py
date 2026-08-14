"""Summarize existing RefCOCO seed-0 testA/testB evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    rows = []
    for split in ("testA", "testB"):
        for variant in ("D0", "A4", "S4", "A8"):
            run = torch.load(args.predictions / f"refcoco-{variant}-s0-{split}.pt", map_location="cpu", weights_only=False)
            rows.append({"split": split, "variant": variant, "n": len(run["ids"]), "iou_at_0_5": float(run["metrics"]["acc_iou_0.5"].float().mean()), "mean_iou": float(run["metrics"]["iou"].float().mean()), "pointing": float(run["metrics"]["pointing"].float().mean()), "target_mass": float(run["metrics"]["target_mass"].float().mean())})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.with_suffix(".csv").write_text("")
    with args.output.with_suffix(".csv").open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# RefCOCO seed-0 held-out evaluation", "", "Existing RefCOCO UNC seed-0 checkpoints only; no new training or seeds. Mass `tau = 0.8`.", "", "| Split | Decoder | N | IoU@0.5 | Mean IoU | Pointing | Target mass |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows: lines.append(f"| {row['split']} | {row['variant']} | {row['n']} | {100*row['iou_at_0_5']:.2f}% | {row['mean_iou']:.4f} | {100*row['pointing']:.2f}% | {row['target_mass']:.4f} |")
    (args.output.with_suffix(".md")).write_text("\n".join(lines)); (args.output.with_suffix(".json")).write_text(json.dumps(rows, indent=2) + "\n"); print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
