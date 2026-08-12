"""Export uniform or train-position-prior predictions without a model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from study import box_target, heatmap_box, per_example_metrics


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("uniform", "position-prior"), required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mass", required=True, type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    records = load(args.data)
    if args.kind == "uniform":
        heatmap = torch.full((576,), 1 / 576)
    else:
        if args.train is None:
            raise ValueError("position-prior requires --train")
        training = load(args.train)
        training_boxes = torch.tensor([record["box_xyxy"] for record in training])
        training_sizes = torch.tensor([[record["width"], record["height"]] for record in training])
        heatmap = box_target(training_boxes, training_sizes).mean(0)
    heatmaps = heatmap.expand(len(records), -1).clone()
    sizes = torch.tensor([[record["width"], record["height"]] for record in records])
    targets = torch.tensor([record["box_xyxy"] for record in records])
    predictions = heatmap_box(heatmaps, sizes, args.mass)
    metrics = per_example_metrics(heatmaps, predictions, targets, sizes)
    result = {
        "ids": [record["id"] for record in records],
        "image_ids": [record["image_id"] for record in records],
        "strata": [record["stratum"] for record in records],
        "tags": [record["tags"] for record in records],
        "compositional": [record["compositional"] for record in records],
        "token_counts": [record["token_count"] for record in records],
        "heatmaps": heatmaps, "predicted_boxes": predictions,
        "target_boxes": targets, "image_sizes": sizes, "metrics": metrics,
        "mass": args.mass, "control": args.kind,
        "manifest_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({name: float(value.float().mean()) for name, value in metrics.items()}))


if __name__ == "__main__":
    main()
