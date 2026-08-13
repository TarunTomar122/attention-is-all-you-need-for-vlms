"""Summarize the locked Ref-Adv-s evaluation and difficulty slices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch


SEED = 20260812
REPLICATES = 10_000
VARIANTS = ("D0", "A4", "S4", "A8")
METRICS = ("acc_iou_0.5", "iou", "pointing", "target_mass")


def quantile_bounds(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    tensor = torch.tensor(ordered, dtype=torch.float64)
    return tuple(float(torch.quantile(tensor, q)) for q in (0.25, 0.5, 0.75))


def quartile_label(value: float, bounds: tuple[float, float, float]) -> str:
    if value <= bounds[0]:
        return "Q1"
    if value <= bounds[1]:
        return "Q2"
    if value <= bounds[2]:
        return "Q3"
    return "Q4"


def load_predictions(paths: dict[str, list[Path]]) -> tuple[list[dict], dict[str, dict[str, torch.Tensor]]]:
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    reference: list[dict] | None = None
    for variant, variant_paths in paths.items():
        runs = [torch.load(path, map_location="cpu", weights_only=False) for path in variant_paths]
        for run in runs:
            current = [
                {"id": identifier, "image_id": image_id, "stratum": stratum,
                 "token_count": int(token_count), "compositional": bool(compositional)}
                for identifier, image_id, stratum, token_count, compositional in zip(
                    run["ids"], run["image_ids"], run["strata"], run["token_counts"], run["compositional"]
                )
            ]
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError(f"prediction files are not paired for {variant}")
        loaded[variant] = {metric: torch.stack([run["metrics"][metric].float() for run in runs]) for metric in METRICS}
    if reference is None or len(reference) != 1142:
        raise ValueError("expected 1,142 paired Ref-Adv examples")
    return reference, loaded


def load_native(path: Path, reference: list[dict]) -> list[dict]:
    native = {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line)}
    if set(native) != {row["id"] for row in reference}:
        raise ValueError("Ref-Adv manifest IDs do not match predictions")
    return [native[row["id"]] for row in reference]


def interval(values: list[float], level: float = 0.95) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    tail = (1 - level) / 2
    return float(torch.quantile(tensor, tail)), float(torch.quantile(tensor, 1 - tail))


def bootstrap_delta(
    difference: torch.Tensor, image_ids: list[str], indices: list[int], *, replicates: int,
) -> tuple[float, tuple[float, float]]:
    observed = float(difference[indices].mean())
    clusters: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        clusters[str(image_ids[index])].append(index)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        raise ValueError("cannot bootstrap an empty slice")
    randomizer = random.Random(SEED)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [randomizer.choice(cluster_ids) for _ in cluster_ids]
        selected_indices = [index for cluster in selected for index in clusters[cluster]]
        samples.append(float(difference[selected_indices].mean()))
    return observed, interval(samples)


def make_slices(reference: list[dict], native: list[dict]) -> tuple[dict[str, list[int]], dict]:
    lengths = [row["token_count"] for row in reference]
    distractors = []
    for row in native:
        try:
            distractors.append(float(row["native"]["distractors"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Ref-Adv distractors must be numeric native metadata") from error
    length_bounds = quantile_bounds([float(value) for value in lengths])
    distractor_bounds = quantile_bounds(distractors)
    slices: dict[str, list[int]] = {"overall": list(range(len(reference)))}
    slices["negation"] = [index for index, row in enumerate(native) if bool(row["native"]["use_negation"])]
    slices["non_negation"] = [index for index, row in enumerate(native) if not bool(row["native"]["use_negation"])]
    for name, values, bounds in (
        ("length", [float(value) for value in lengths], length_bounds),
        ("distractors", distractors, distractor_bounds),
    ):
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            slices[f"{name}_{quartile}"] = [
                index for index, value in enumerate(values) if quartile_label(value, bounds) == quartile
            ]
    rules = {
        "seed": SEED,
        "replicates": REPLICATES,
        "length": {"method": "inclusive empirical quartiles", "q1_q2_q3": length_bounds},
        "distractors": {"method": "inclusive empirical quartiles", "q1_q2_q3": distractor_bounds},
        "official_metadata": ["use_negation", "distractors", "image_source", "human_authored"],
        "official_reasoning_facets": "not present in the prepared Ref-Adv-s schema; no semantic labels were invented",
    }
    return slices, rules


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    groups = (("length", "Expression length quartile"), ("distractors", "Distractor-count quartile"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (prefix, title) in zip(axes, groups):
        selected = [row for row in rows if row["slice"].startswith(prefix + "_")]
        selected.sort(key=lambda row: int(row["slice"][-1]))
        x = range(1, len(selected) + 1)
        for variant, color in (("A4", "#1f77b4"), ("S4", "#d62728"), ("A8", "#2ca02c")):
            axis.plot(x, [100 * row[variant] for row in selected], marker="o", label=variant, color=color)
        axis.set_title(title)
        axis.set_xlabel("Quartile (dataset-defined)")
        axis.set_ylabel("IoU@0.5 (%)")
        axis.set_xticks(list(x), ["Q1", "Q2", "Q3", "Q4"])
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(out / "refadv_performance_by_difficulty.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (prefix, title) in zip(axes, groups):
        selected = [row for row in rows if row["slice"].startswith(prefix + "_")]
        selected.sort(key=lambda row: int(row["slice"][-1]))
        x = list(range(1, len(selected) + 1))
        for delta, color, label, low, high in (
            ("delta_a4_s4", "#1f77b4", "A4 − S4", "a4_s4_ci95_low", "a4_s4_ci95_high"),
            ("delta_a8_s4", "#2ca02c", "A8 − S4", "a8_s4_ci95_low", "a8_s4_ci95_high"),
        ):
            y = [100 * row[delta] for row in selected]
            error = [[100 * (row[delta] - row[low]) for row in selected],
                     [100 * (row[high] - row[delta]) for row in selected]]
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=label, color=color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Quartile (dataset-defined)")
        axis.set_ylabel("Delta IoU@0.5 (percentage points)")
        axis.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(out / "refadv_delta_by_difficulty.png", dpi=180)
    plt.close(fig)


def classify(rows: list[dict]) -> tuple[str, str]:
    hard = [row for row in rows if row["slice"] in {"length_Q4", "distractors_Q4", "negation"} and row["n"] >= 30]
    clear_loss = [row for row in hard if row["a4_s4_ci95_high"] < 0]
    if not clear_loss:
        practical_match = all(row["a4_s4_ci95_low"] > -0.05 for row in hard)
        if practical_match:
            return "Case A", "A4 remains within the practical margin on every sufficiently populated hard slice."
        return "Case D", "The slice estimates are too noisy or mixed to establish a reliable failure boundary."
    recovered = [row for row in clear_loss if row["a8_s4"] > row["a4_s4"] and row["a8_s4"] > -0.05]
    if recovered:
        return "Case B", "A4 has a clear hard-slice deficit and A8 recovers at least the practical gap on a hard slice."
    return "Case C", "A4 has a clear hard-slice deficit and A8 does not recover it in the available evidence."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        variant: [args.predictions / f"siglip2-{variant}-s{seed}.pt" for seed in (0, 1, 2)]
        for variant in VARIANTS
    }
    reference, loaded = load_predictions(paths)
    native = load_native(args.data, reference)
    slices, rules = make_slices(reference, native)
    averaged = {variant: {metric: values.mean(0) for metric, values in metrics.items()} for variant, metrics in loaded.items()}
    differences = {
        "a4_s4": averaged["A4"]["acc_iou_0.5"] - averaged["S4"]["acc_iou_0.5"],
        "a8_s4": averaged["A8"]["acc_iou_0.5"] - averaged["S4"]["acc_iou_0.5"],
    }
    slice_rows = []
    bootstrap = {"protocol": rules, "slices": {}}
    for name, indices in slices.items():
        row = {"slice": name, "n": len(indices)}
        for variant in ("A4", "S4", "A8"):
            row[variant] = float(averaged[variant]["acc_iou_0.5"][indices].mean())
        row["delta_a4_s4"] = row["A4"] - row["S4"]
        row["delta_a8_s4"] = row["A8"] - row["S4"]
        a4_observed, a4_ci = bootstrap_delta(differences["a4_s4"], [row["image_id"] for row in reference], indices, replicates=REPLICATES)
        a8_observed, a8_ci = bootstrap_delta(differences["a8_s4"], [row["image_id"] for row in reference], indices, replicates=REPLICATES)
        row.update({"a4_s4_ci95_low": a4_ci[0], "a4_s4_ci95_high": a4_ci[1],
                    "a8_s4_ci95_low": a8_ci[0], "a8_s4_ci95_high": a8_ci[1]})
        slice_rows.append(row)
        bootstrap["slices"][name] = {
            "n": len(indices), "a4_s4": a4_observed, "a4_s4_ci95": list(a4_ci),
            "a8_s4": a8_observed, "a8_s4_ci95": list(a8_ci),
        }

    per_example = []
    for index, (record, source) in enumerate(zip(reference, native)):
        row = {"id": record["id"], "image_id": record["image_id"], "expression": source["expression"],
               "use_negation": bool(source["native"]["use_negation"]),
               "distractors": int(source["native"]["distractors"]), "image_source": source["native"]["image_source"],
               "human_authored": bool(source["native"]["human_authored"]), "token_count": record["token_count"],
               "stratum": record["stratum"]}
        for variant in VARIANTS:
            for metric in METRICS:
                row[f"{variant}_{metric}"] = float(averaged[variant][metric][index])
        row["A4_minus_S4"] = float(differences["a4_s4"][index])
        row["A8_minus_S4"] = float(differences["a8_s4"][index])
        per_example.append(row)

    write_csv(args.output / "refadv_per_example.csv", per_example)
    write_csv(args.output / "refadv_slices.csv", slice_rows)
    (args.output / "refadv_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    plot_results(args.output, slice_rows)
    case, reason = classify(slice_rows)
    overall = next(row for row in slice_rows if row["slice"] == "overall")
    summary = [
        "# Ref-Adv-s failure-boundary summary", "",
        "Protocol: pinned Ref-Adv-s revision, 1,142 test examples, frozen `tau = 0.8`, SigLIP2 RefCOCOg checkpoints, three seeds, and image-clustered bootstrap (10,000 replicates; seed 20260812).", "",
        "## Overall IoU@0.5", "",
        "| Model | Accuracy |", "| --- | ---: |",
        f"| A4 | {100 * overall['A4']:.2f}% |", f"| S4 | {100 * overall['S4']:.2f}% |", f"| A8 | {100 * overall['A8']:.2f}% |",
        f"| A4 − S4 | {100 * overall['delta_a4_s4']:+.2f} pp |", f"| A8 − S4 | {100 * overall['delta_a8_s4']:+.2f} pp |", "",
        "## Slice results", "", "See `refadv_slices.csv` for every slice, confidence interval, and sample count. Bins are inclusive empirical quartiles computed from Ref-Adv metadata before model-performance slicing.", "",
        "| Slice | N | A4 | S4 | A8 | A4−S4 | A8−S4 | A4−S4 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in slice_rows:
        summary.append(f"| {row['slice']} | {row['n']} | {100*row['A4']:.2f}% | {100*row['S4']:.2f}% | {100*row['A8']:.2f}% | {100*row['delta_a4_s4']:+.2f} pp | {100*row['delta_a8_s4']:+.2f} pp | [{100*row['a4_s4_ci95_low']:+.2f}, {100*row['a4_s4_ci95_high']:+.2f}] pp |")
    summary += ["", "## Interpretation gate", "", f"**{case}.** {reason}", "", "Official reasoning/facet annotations were not present in the prepared schema, so no semantic labels were invented. The native fields used here are negation, distractor count, image source, and human-authored status.", ""]
    (args.output / "refadv_summary.md").write_text("\n".join(summary))
    (args.output / "refadv_interpretation.md").write_text(
        f"# Ref-Adv-s interpretation\n\n**Classification: {case}.** {reason}\n\n"
        f"A4 overall: {100*overall['A4']:.2f}%; S4 overall: {100*overall['S4']:.2f}%; A8 overall: {100*overall['A8']:.2f}%. "
        f"Overall A4−S4: {100*overall['delta_a4_s4']:+.2f} percentage points, with 95% CI "
        f"[{100*overall['a4_s4_ci95_low']:+.2f}, {100*overall['a4_s4_ci95_high']:+.2f}].\n\n"
        "Read the length and distractor quartile plots together: a boundary requires a consistent worsening A4−S4 trend, not one isolated slice. "
        "A8 recovery is evidence about attention capacity, not proof that the FFN itself is causal. The next expensive experiment is not launched automatically.\n"
    )
    print(json.dumps({"case": case, "overall": overall, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
