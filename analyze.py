"""Run the preregistered paired, image-clustered confirmatory analysis."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def load(paths: list[Path]) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    runs = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    for key in ("ids", "image_ids", "strata"):
        if any(run[key] != runs[0][key] for run in runs[1:]):
            raise ValueError(f"paired prediction files disagree on {key}")
    values = torch.stack([run["metrics"]["acc_iou_0.5"].float() for run in runs])
    return runs[0]["ids"], runs[0]["image_ids"], runs[0]["strata"], values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", nargs=3, required=True, type=Path)
    parser.add_argument("--standard", nargs=3, required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    ids, image_ids, strata, attention = load(args.attention)
    standard_ids, standard_image_ids, standard_strata, standard = load(args.standard)
    if (ids, image_ids, strata) != (standard_ids, standard_image_ids, standard_strata):
        raise ValueError("attention and standard examples are not paired")
    difference = (attention - standard).mean(0)
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, image_id in enumerate(image_ids):
        clusters[str(image_id)].append(index)
    cluster_ids = sorted(clusters)
    randomizer = random.Random(20260812)

    def means(indices: list[int]) -> dict[str, float]:
        return {
            stratum: float(difference[[i for i in indices if strata[i] == stratum]].mean())
            for stratum in sorted(set(strata))
            if any(strata[i] == stratum for i in indices)
        }

    observed = means(list(range(len(ids))))
    samples = {name: [] for name in observed}
    interactions = []
    for _ in range(args.replicates):
        selected = [randomizer.choice(cluster_ids) for _ in cluster_ids]
        indices = [index for cluster in selected for index in clusters[cluster]]
        replicate = means(indices)
        for name, value in replicate.items():
            samples[name].append(value)
        interactions.append(replicate["direct"] - replicate["logical"])

    def interval(values: list[float], level: float) -> list[float]:
        tail = (1 - level) / 2
        tensor = torch.tensor(values)
        return [float(torch.quantile(tensor, tail)), float(torch.quantile(tensor, 1 - tail))]

    interaction = observed["direct"] - observed["logical"]
    direct_90 = interval(samples["direct"], 0.90)
    interaction_95 = interval(interactions, 0.95)
    output = {
        "per_seed_delta_accuracy": {
            f"seed_{seed}": {
                stratum: float((attention[seed] - standard[seed])[
                    [index for index, value in enumerate(strata) if value == stratum]
                ].mean())
                for stratum in sorted(set(strata))
            }
            for seed in range(3)
        },
        "delta_accuracy": observed,
        "delta_95_ci": {name: interval(values, 0.95) for name, values in samples.items()},
        "direct_90_ci": direct_90,
        "direct_minus_logical": interaction,
        "interaction_95_ci": interaction_95,
        "conditions": {
            "direct_retention": direct_90[0] > -0.05,
            "task_interaction": interaction >= 0.05 and interaction_95[0] > 0,
        },
    }
    output["main_claim_supported"] = all(output["conditions"].values())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
