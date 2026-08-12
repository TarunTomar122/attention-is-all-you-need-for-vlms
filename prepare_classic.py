"""Normalize official RefCOCO-family annotations into deterministic JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

from study import classify_expression


def prepare(
    dataset: str, split: str, instances_path: Path, refs_path: Path,
    image_dir: Path, output_path: Path,
) -> tuple[int, str, list[str]]:
    instances = json.loads(instances_path.read_text())
    with refs_path.open("rb") as file:
        refs = pickle.load(file)
    images = {item["id"]: item for item in instances["images"]}
    annotations = {item["id"]: item for item in instances["annotations"]}
    records: list[dict] = []
    seen: set[str] = set()
    clamped: list[str] = []

    for ref in refs:
        if ref["split"] != split:
            continue
        image = images[ref["image_id"]]
        annotation = annotations[ref["ann_id"]]
        image_path = image_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        x, y, width, height = map(float, annotation["bbox"])
        raw_box = (x, y, x + width, y + height)
        if not all(math.isfinite(value) for value in raw_box) or width <= 0 or height <= 0:
            raise ValueError(f"invalid box for annotation {ref['ann_id']}")
        limits = (0.0, 0.0, float(image["width"]), float(image["height"]))
        overflow = max(lower - value for lower, value in zip(limits[:2], raw_box[:2]))
        overflow = max(overflow, *(value - upper for value, upper in zip(raw_box[2:], limits[2:])))
        if overflow >= 1.0:
            raise ValueError(f"box exceeds image by {overflow:.3f}px for annotation {ref['ann_id']}")
        box = [
            min(max(raw_box[0], 0.0), limits[2]),
            min(max(raw_box[1], 0.0), limits[3]),
            min(max(raw_box[2], 0.0), limits[2]),
            min(max(raw_box[3], 0.0), limits[3]),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"box is outside image for annotation {ref['ann_id']}")
        was_clamped = tuple(box) != raw_box

        for sentence in ref["sentences"]:
            identifier = f"{dataset}:{sentence['sent_id']}"
            if identifier in seen:
                raise ValueError(f"duplicate expression identifier: {identifier}")
            seen.add(identifier)
            expression = sentence["sent"].strip()
            if not expression:
                raise ValueError(f"empty expression: {identifier}")
            classification = classify_expression(expression)
            records.append({
                "id": identifier,
                "dataset": dataset,
                "split": split,
                "image": str(image_path.resolve()),
                "image_id": int(ref["image_id"]),
                "category_id": int(ref["category_id"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "expression": expression,
                "box_xyxy": box,
                "tags": list(classification.tags),
                "stratum": classification.stratum,
                "compositional": classification.compositional,
                "token_count": classification.token_count,
            })
            if was_clamped:
                clamped.append(identifier)

    records.sort(key=lambda record: record["id"])
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    payload = ("\n".join(lines) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return len(records), hashlib.sha256(payload).hexdigest(), clamped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("refcoco", "refcoco+", "refcocog"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--refs", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count, digest, clamped = prepare(
        args.dataset, args.split, args.instances, args.refs, args.images, args.output,
    )
    print(json.dumps({"examples": count, "sha256": digest, "clamped_ids": clamped}))


if __name__ == "__main__":
    main()
