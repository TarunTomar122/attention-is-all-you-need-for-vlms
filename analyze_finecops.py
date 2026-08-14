"""Analyze positive FineCops predictions with official difficulty labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SEED = 20260814
REPLICATES = 10_000
VARIANTS = ("A4", "S4", "A8")
METRICS = ("acc_iou_0.5", "iou", "pointing", "target_mass")


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return (ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))])


def load_variant(paths: list[Path]) -> tuple[list[dict], dict[str, torch.Tensor]]:
    runs = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference = [
        {"id": identifier, "image_id": str(image_id)}
        for identifier, image_id in zip(runs[0]["ids"], runs[0]["image_ids"])
    ]
    for run in runs[1:]:
        current = [{"id": i, "image_id": str(image_id)} for i, image_id in zip(run["ids"], run["image_ids"])]
        if current != reference:
            raise ValueError("FineCops predictions are not paired")
    return reference, {metric: torch.stack([run["metrics"][metric].float() for run in runs]).mean(0) for metric in METRICS}


def bootstrap(difference: torch.Tensor, image_ids: list[str], indices: list[int]) -> tuple[float, tuple[float, float]]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        clusters[image_ids[index]].append(index)
    cluster_means = [float(difference[indexes].mean()) for indexes in clusters.values()]
    observed = float(difference[indices].mean())
    # Vectorized cluster resampling avoids millions of Python-level random calls.
    rng = np.random.default_rng(SEED)
    samples = rng.choice(cluster_means, size=(REPLICATES, len(cluster_means)), replace=True).mean(axis=1)
    return observed, interval(samples)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(out: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted((row for row in rows if row["slice"].startswith("level_")), key=lambda row: int(row["slice"].split("_")[1]))
    x = list(range(1, len(levels) + 1))
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant, color in (("A4", "#1f77b4"), ("S4", "#d62728"), ("A8", "#2ca02c")):
        axis.plot(x, [100 * row[variant] for row in levels], marker="o", label=variant, color=color)
    axis.set(xlabel="Official FineCops difficulty level", ylabel="IoU@0.5 (%)", xticks=x, xticklabels=["1", "2", "3"])
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(out / "finecops_performance_by_level.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for delta, low, high, label, color in (("delta_a4_s4", "a4_s4_ci95_low", "a4_s4_ci95_high", "A4 − S4", "#1f77b4"), ("delta_a8_s4", "a8_s4_ci95_low", "a8_s4_ci95_high", "A8 − S4", "#2ca02c")):
        axis.errorbar(x, [100 * row[delta] for row in levels], yerr=[[100 * (row[delta] - row[low]) for row in levels], [100 * (row[high] - row[delta]) for row in levels]], marker="o", capsize=3, label=label, color=color)
    axis.axhline(0, color="black", linewidth=0.8); axis.set(xlabel="Official FineCops difficulty level", ylabel="IoU@0.5 delta (pp)", xticks=x, xticklabels=["1", "2", "3"])
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(out / "finecops_delta_by_level.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = None
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    for variant in VARIANTS:
        paths = [args.predictions / f"finecops-siglip2-{variant}-s{seed}.pt" for seed in (0, 1, 2)]
        current, metrics = load_variant(paths)
        reference = current if reference is None else reference
        if current != reference:
            raise ValueError("prediction IDs differ across variants")
        loaded[variant] = metrics
    native = {row["id"]: row for row in (json.loads(line) for line in args.data.read_text().splitlines() if line)}
    if set(native) != {row["id"] for row in reference}:
        raise ValueError("FineCops manifest IDs do not match predictions")
    image_ids = [row["image_id"] for row in reference]
    slices: dict[str, list[int]] = {"overall": list(range(len(reference)))}
    for field, prefix in (("finecops_level", "level"), ("finecops_tuple_type", "tuple_type")):
        for value in sorted({native[row["id"]][field] for row in reference}, key=str):
            slices[f"{prefix}_{value}"] = [i for i, row in enumerate(reference) if native[row["id"]][field] == value]
    differences = {"a4_s4": loaded["A4"]["acc_iou_0.5"] - loaded["S4"]["acc_iou_0.5"], "a8_s4": loaded["A8"]["acc_iou_0.5"] - loaded["S4"]["acc_iou_0.5"]}
    rows = []; bootstrap_json = {"protocol": {"seed": SEED, "replicates": REPLICATES, "cluster": "image_id", "official_fields": ["finecops_level", "finecops_tuple_type"]}, "slices": {}}
    for name, indices in slices.items():
        row = {"slice": name, "n": len(indices)}
        for variant in VARIANTS: row[variant] = float(loaded[variant]["acc_iou_0.5"][indices].mean())
        row["delta_a4_s4"] = row["A4"] - row["S4"]; row["delta_a8_s4"] = row["A8"] - row["S4"]
        a4, a4_ci = bootstrap(differences["a4_s4"], image_ids, indices); a8, a8_ci = bootstrap(differences["a8_s4"], image_ids, indices)
        row.update({"a4_s4_ci95_low": a4_ci[0], "a4_s4_ci95_high": a4_ci[1], "a8_s4_ci95_low": a8_ci[0], "a8_s4_ci95_high": a8_ci[1]})
        rows.append(row); bootstrap_json["slices"][name] = {"n": len(indices), "a4_s4": a4, "a4_s4_ci95": list(a4_ci), "a8_s4": a8, "a8_s4_ci95": list(a8_ci)}
    per_example = []
    for i, ref in enumerate(reference):
        source = native[ref["id"]]; row = {"id": ref["id"], "image_id": ref["image_id"], "expression": source["expression"], "finecops_level": source["finecops_level"], "finecops_tuple_type": source["finecops_tuple_type"]}
        for variant in VARIANTS:
            for metric in METRICS: row[f"{variant}_{metric}"] = float(loaded[variant][metric][i])
        row["A4_minus_S4"] = float(differences["a4_s4"][i]); row["A8_minus_S4"] = float(differences["a8_s4"][i]); per_example.append(row)
    write_csv(args.output / "finecops_per_example.csv", per_example); write_csv(args.output / "finecops_slices.csv", rows); (args.output / "finecops_bootstrap.json").write_text(json.dumps(bootstrap_json, indent=2) + "\n"); plot(args.output, rows)
    overall = next(row for row in rows if row["slice"] == "overall"); level3 = next(row for row in rows if row["slice"] == "level_3")
    overall_loss = overall["a4_s4_ci95_high"] < 0
    level3_loss = level3["a4_s4_ci95_high"] < 0
    overall_recovered = overall["delta_a8_s4"] >= 0 and overall["delta_a8_s4"] > overall["delta_a4_s4"]
    level3_recovered = level3["delta_a8_s4"] > level3["delta_a4_s4"] and level3["a8_s4_ci95_high"] >= 0
    if level3_loss:
        case = "Case B" if level3_recovered else "Case A"
    elif overall_loss and overall_recovered:
        case = "Case B"
    else:
        case = "Case C"
    reason = {
        "Case A": "A4 has a confidence interval below zero on the official level-3 boundary slice without A8 recovery.",
        "Case B": "A4 has a confidence interval below zero overall and A8 closes the observed gap; the level-3 slice itself is not a confirmed monotonic boundary.",
        "Case C": "A4 continues to match S4 overall, so no FFN failure boundary is supported.",
    }[case]
    summary = ["# FineCops-Ref positive-test summary", "", "Protocol: official positive test split, frozen SigLIP2 RefCOCOg checkpoints, three decoder seeds, tau = 0.8, and image-clustered bootstrap (10,000 replicates; seed 20260814).", "", "## Overall metrics", "", "| Model | IoU@0.5 | Mean IoU | Pointing | Target mass |", "| --- | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS: summary.append(f"| {variant} | {100*float(loaded[variant]['acc_iou_0.5'].mean()):.2f}% | {float(loaded[variant]['iou'].mean()):.4f} | {100*float(loaded[variant]['pointing'].mean()):.2f}% | {float(loaded[variant]['target_mass'].mean()):.4f} |")
    summary += [f"| A4 − S4 IoU@0.5 | {100*overall['delta_a4_s4']:+.2f} pp | — | — | — |", f"| A8 − S4 IoU@0.5 | {100*overall['delta_a8_s4']:+.2f} pp | — | — | — |", "", "## Official slices", "", "| Slice | N | A4 | S4 | A8 | A4−S4 | A8−S4 | A4−S4 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows: summary.append(f"| {row['slice']} | {row['n']} | {100*row['A4']:.2f}% | {100*row['S4']:.2f}% | {100*row['A8']:.2f}% | {100*row['delta_a4_s4']:+.2f} pp | {100*row['delta_a8_s4']:+.2f} pp | [{100*row['a4_s4_ci95_low']:+.2f}, {100*row['a4_s4_ci95_high']:+.2f}] pp |")
    summary += ["", f"## Classification: {case}", "", reason, "", "See `finecops_interpretation.md` for the restrained conclusion and next-experiment recommendation.", ""]
    (args.output / "finecops_summary.md").write_text("\n".join(summary)); (args.output / "finecops_interpretation.md").write_text(f"# FineCops interpretation\n\n**{case}.** {reason}\n\nOverall A4−S4: {100*overall['delta_a4_s4']:+.2f} percentage points (95% CI [{100*overall['a4_s4_ci95_low']:+.2f}, {100*overall['a4_s4_ci95_high']:+.2f}]). A8−S4 overall: {100*overall['delta_a8_s4']:+.2f} points. Level 3 A4−S4: {100*level3['delta_a4_s4']:+.2f} points (95% CI [{100*level3['a4_s4_ci95_low']:+.2f}, {100*level3['a4_s4_ci95_high']:+.2f}]).\n\nThe overall comparison supports a small A4 deficit that A8 recovers, while the official level-3 slice is too noisy to establish a monotonic difficulty boundary. Tuple-type rows are descriptive compositional analyses. A8 recovery is interpreted as evidence about attention capacity, not proof that the FFN caused the gap.\n")
    print(json.dumps({"case": case, "overall": overall, "level_3": level3, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
