"""Prepare the official FineCops positive test split as study JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study import WORDS


def prepare(annotation_path: Path, expression_path: Path, image_dir: Path, output: Path) -> tuple[int, str, int]:
    coco = json.loads(annotation_path.read_text())
    expressions = json.loads(expression_path.read_text())
    images = {int(row["id"]): row for row in coco["images"]}
    annotations = {int(row["image_id"]): row for row in coco["annotations"]}
    records: list[dict] = []
    boundary_clamped = 0
    for key, expression_row in expressions.items():
        image_id = int(expression_row["id"])
        image = images[image_id]
        annotation = annotations[image_id]
        expression = str(expression_row["expression"]).strip()
        if expression != str(image["caption"]).strip():
            raise ValueError(f"caption mismatch for FineCops image {image_id}")
        image_path = image_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        x, y, width, height = map(float, annotation["bbox"])
        raw_box = [x, y, x + width, y + height]
        overflow = max(
            0.0,
            -raw_box[0],
            -raw_box[1],
            raw_box[2] - image["width"],
            raw_box[3] - image["height"],
        )
        if width <= 0 or height <= 0 or overflow > 2.0:
            raise ValueError(f"invalid FineCops box for image {image_id}: {raw_box}")
        box = [
            max(0.0, min(raw_box[0], float(image["width"]))),
            max(0.0, min(raw_box[1], float(image["height"]))),
            max(0.0, min(raw_box[2], float(image["width"]))),
            max(0.0, min(raw_box[3], float(image["height"]))),
        ]
        if box != raw_box:
            boundary_clamped += 1
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError(f"invalid FineCops box for image {image_id}: {raw_box}")
        level = int(annotation["level"])
        tuple_type = str(annotation["tuple_type"])
        records.append({
            "id": f"finecops:test:{image_id}",
            "dataset": "finecops",
            "split": "test-positive",
            "image": str(image_path.resolve()),
            "image_id": str(expression_row["image_id"]),
            "category_id": int(annotation["category_id"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "expression": expression,
            "box_xyxy": box,
            "box_boundary_clamped": box != raw_box,
            "tags": [f"tuple_type:{tuple_type}"],
            "stratum": f"level_{level}",
            "compositional": level >= 2,
            "token_count": len(WORDS.findall(expression.lower())),
            "finecops_level": level,
            "finecops_tuple_type": tuple_type,
            "finecops_spatial": annotation.get("spatial", ""),
            "finecops_attribute": annotation.get("attribute", []),
            "finecops_objects_id": annotation.get("objects_id", []),
            "finecops_tuple": annotation.get("tuple", []),
        })
    records.sort(key=lambda row: int(row["image_id"]))
    payload = ("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return len(records), hashlib.sha256(payload).hexdigest(), boundary_clamped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--expressions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    count, digest, boundary_clamped = prepare(args.annotations, args.expressions, args.images, args.output)
    print(json.dumps({"examples": count, "sha256": digest, "boundary_clamped": boundary_clamped}))


if __name__ == "__main__":
    main()
