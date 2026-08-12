"""Select the one global heatmap-to-box mass on S4 RefCOCOg validation."""

import argparse
import json
from pathlib import Path

import torch

from study import heatmap_box, per_example_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    result = torch.load(args.predictions, map_location="cpu", weights_only=False)
    scores = {}
    for mass in (0.5, 0.6, 0.7, 0.8, 0.9):
        boxes = heatmap_box(result["heatmaps"].float(), result["image_sizes"], mass)
        metrics = per_example_metrics(
            result["heatmaps"].float(), boxes, result["target_boxes"], result["image_sizes"],
        )
        scores[str(mass)] = float(metrics["acc_iou_0.5"].float().mean())
    best = max(scores, key=lambda mass: (scores[mass], -float(mass)))
    print(json.dumps({"selected_mass": float(best), "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
