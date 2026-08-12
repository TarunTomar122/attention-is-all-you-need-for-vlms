"""Materialize the pinned Ref-Adv-s release as test-only JSONL."""

import argparse
import json
import math
from pathlib import Path

from study import classify_expression


DATASET = "dddraxxx/ref-adv-s"
REVISION = "e7a53e352b5885b8228fc6afa8645ab78e76d5f1"


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    image_dir = args.output.parent / "refadv-images"
    image_dir.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(DATASET, split="train", revision=REVISION)
    lines = []
    for index, row in enumerate(dataset):
        expression = row["normal_caption"].strip()
        box = list(map(float, row["solution"]))
        width, height = int(row["width"]), int(row["height"])
        if (
            len(box) != 4 or not all(map(math.isfinite, box))
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
        ):
            raise ValueError(f"invalid Ref-Adv-s box at row {index}")
        image_path = image_dir / f"{index:04}.jpg"
        row["image"].convert("RGB").save(image_path, quality=95)
        classification = classify_expression(expression)
        lines.append(json.dumps({
            "id": f"refadv:{index}", "dataset": "refadv", "split": "test",
            "image": str(image_path.resolve()), "image_id": f"refadv:{index}",
            "width": width, "height": height, "expression": expression,
            "box_xyxy": box, "tags": list(classification.tags),
            "stratum": classification.stratum,
            "compositional": classification.compositional,
            "token_count": classification.token_count,
            "native": {
                "distractors": row["distractors"],
                "use_negation": row["use_negation"],
                "image_source": row["image_source"],
                "human_authored": row["human_authored"],
            },
        }, sort_keys=True, separators=(",", ":")))
    if len(lines) != 1142:
        raise ValueError(f"expected 1142 Ref-Adv-s rows, received {len(lines)}")
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} test rows to {args.output}")


if __name__ == "__main__":
    main()
