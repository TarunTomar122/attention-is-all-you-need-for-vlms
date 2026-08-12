"""Sample the frozen training-only taxonomy audit as a reviewable CSV."""

import argparse
import csv
import pickle
import random
import shutil
from pathlib import Path

from study import classify_expression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-stratum", type=int, default=40)
    args = parser.parse_args()
    with args.refs.open("rb") as file:
        refs = pickle.load(file)
    buckets = {name: [] for name in ("direct", "absolute", "relational", "logical", "unclassified")}
    for ref in refs:
        if ref["split"] != "train":
            continue
        for sentence in ref["sentences"]:
            result = classify_expression(sentence["sent"])
            buckets[result.stratum].append((sentence["sent_id"], sentence["sent"], result))
    randomizer = random.Random(20260812)
    rows = []
    for stratum, candidates in buckets.items():
        for identifier, expression, result in randomizer.sample(candidates, args.per_stratum):
            rows.append({
                "sent_id": identifier, "expression": expression,
                "predicted_stratum": stratum, "predicted_tags": "|".join(result.tags),
                "predicted_compositional": result.compositional,
                "review": "", "review_note": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        frozen = args.output.with_name(f"{args.output.stem}-training-audit-v1{args.output.suffix}")
        shutil.copyfile(args.output, frozen)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
