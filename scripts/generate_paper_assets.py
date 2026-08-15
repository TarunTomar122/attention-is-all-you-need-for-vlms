#!/usr/bin/env python3
"""Generate paper figures and tables from the frozen, versioned result artifacts."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FINECOPS = ROOT / "docs/results/finecops"
REFADV = ROOT / "docs/results/refadv"
EFFICIENCY = ROOT / "docs/results/efficiency"
CLIP = ROOT / "docs/results/clip-control"
REFCOCOG = ROOT / "docs/results/refcocog-three-seed-summary.md"
COLORS = {"ink": "#17211b", "blue": "#2563eb", "red": "#dc2626", "green": "#15803d", "muted": "#64748b", "grid": "#dbe3ec"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, stem: str) -> None:
    out = PAPER / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": dt.datetime(1970, 1, 1, tzinfo=dt.UTC), "ModDate": dt.datetime(1970, 1, 1, tzinfo=dt.UTC)} if suffix == "pdf" else None
        fig.savefig(out / f"generated-{stem}.{suffix}", dpi=220 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
    plt.close(fig)


def parse_pp(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}[^+−-]*([+−-]\d+\.\d+) pp", text)
    if not match:
        raise ValueError(f"Could not find {label!r}")
    return float(match.group(1).replace("−", "-"))


def build() -> dict:
    with (FINECOPS / "finecops_bootstrap.json").open() as handle:
        finecops_bootstrap = json.load(handle)
    with (REFADV / "refadv_bootstrap.json").open() as handle:
        refadv_bootstrap = json.load(handle)
    with (EFFICIENCY / "measurements.json").open() as handle:
        efficiency = json.load(handle)
    with (CLIP / "clip_control_summary.json").open() as handle:
        clip = json.load(handle)
    with (FINECOPS / "finecops_slices.csv").open(newline="") as handle:
        finecops_slices = list(csv.DictReader(handle))

    fine = finecops_bootstrap["slices"]["overall"]
    adv = refadv_bootstrap["slices"]["overall"]
    refcocog = REFCOCOG.read_text()
    return {
        "schema_version": 1,
        "title": "Do Visual Grounding Decoders Need Feed-Forward Networks?",
        "claim_boundary": "FFN-free trainable grounding decoders over frozen VLM features; not fully attention-only VLMs.",
        "frozen_contract": {"tau": 0.8, "bootstrap_seed": finecops_bootstrap["protocol"]["seed"], "bootstrap_replicates": finecops_bootstrap["protocol"]["replicates"]},
        "results": {
            "refcocog_direct_a4_minus_s4_pp": parse_pp(refcocog, "A4 − S4, direct"),
            "refadv": {"a4_minus_s4_pp": 100 * adv["a4_s4"], "ci95_pp": [100 * value for value in adv["a4_s4_ci95"]]},
            "finecops": {"a4_minus_s4_pp": 100 * fine["a4_s4"], "ci95_pp": [100 * value for value in fine["a4_s4_ci95"]], "a8_minus_s4_pp": 100 * fine["a8_s4"]},
            "clip_one_seed_a4_minus_s4_pp": 100 * clip["metrics"]["acc_iou_0.5"]["A4_minus_S4"],
            "efficiency": efficiency,
        },
        "finecops_slices": finecops_slices,
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (FINECOPS / "finecops_bootstrap.json", FINECOPS / "finecops_slices.csv", REFADV / "refadv_bootstrap.json", EFFICIENCY / "measurements.json", CLIP / "clip_control_summary.json", REFCOCOG)
        ],
    }


def write_table(data: dict) -> None:
    fine = data["results"]["finecops"]
    adv = data["results"]["refadv"]
    rows = [
        ["RefCOCOg direct (3 seeds)", f"{data['results']['refcocog_direct_a4_minus_s4_pp']:+.2f}", "90% CI [-0.33, +0.84]", "Direct-retention gate passed; interaction not confirmed"],
        ["Ref-Adv-s overall (3 seeds)", f"{adv['a4_minus_s4_pp']:+.2f}", f"95% CI [{adv['ci95_pp'][0]:+.2f}, {adv['ci95_pp'][1]:+.2f}]", "No monotonic hard-slice boundary"],
        ["FineCops-Ref overall (3 seeds)", f"{fine['a4_minus_s4_pp']:+.2f}", f"95% CI [{fine['ci95_pp'][0]:+.2f}, {fine['ci95_pp'][1]:+.2f}]", f"A8-S4 = {fine['a8_minus_s4_pp']:+.2f} pp"],
        ["CLIP RefCOCOg (1 seed)", f"{data['results']['clip_one_seed_a4_minus_s4_pp']:+.2f}", "descriptive only", "Second frozen backbone"],
    ]
    out = PAPER / "tables"
    out.mkdir(parents=True, exist_ok=True)
    headers = ["Evaluation", "A4-S4 IoU@0.5 (pp)", "Interval", "Reading"]
    with (out / "generated-main-results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(headers); writer.writerows(rows)
    markdown = ["| " + " | ".join(headers) + " |", "| --- | ---: | --- | --- |"] + ["| " + " | ".join(row) + " |" for row in rows]
    (out / "generated-main-results.md").write_text("\n".join(markdown) + "\n")
    latex_rows = ["\\begin{tabular}{p{0.27\\linewidth}rll}", "\\toprule", "Evaluation & A4--S4 (pp) & Interval & Reading \\\\", "\\midrule"]
    latex_rows += [" & ".join(value.replace("%", "\\%") for value in row) + " \\\\" for row in rows]
    latex_rows += ["\\bottomrule", "\\end{tabular}"]
    (out / "generated-main-results.tex").write_text("\n".join(latex_rows) + "\n")


def method_overview() -> None:
    fig, axis = plt.subplots(figsize=(10.8, 3.4)); axis.set_axis_off(); axis.set(xlim=(0, 13), ylim=(0, 4.2))
    def box(x: float, y: float, width: float, height: float, label: str, color: str = "#f8fafc") -> None:
        axis.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor=COLORS["ink"], linewidth=1.1))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.7, wrap=True)
    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.1, color=COLORS["muted"], shrinkA=3, shrinkB=6))
    box(.3, 1.45, 1.5, 1.2, "Image\n+ expression")
    box(2.2, 1.45, 1.8, 1.2, "Frozen VLM\nfeatures", "#eef6ff")
    box(4.8, 2.55, 2.3, .95, "A4\n4 attention blocks", "#eaf3ff")
    box(4.8, 1.35, 2.3, .95, "S4\n4 attention + FFN blocks", "#fff0f0")
    box(4.8, .15, 2.3, .95, "A8\n8 attention blocks", "#ebfaef")
    box(8.3, 1.45, 1.9, 1.2, "Shared patch\nreadout", "#f8fafc")
    box(10.9, 1.45, 1.7, 1.2, "Patch heatmap\nto box")
    arrow(1.8, 2.05, 2.2, 2.05)
    for y in (3.025, 1.825, .625): arrow(4.0, 2.05, 4.8, y)
    for y in (3.025, 1.825, .625): arrow(7.1, y, 8.3, 2.05)
    arrow(10.2, 2.05, 10.9, 2.05)
    axis.text(6.45, 3.92, "Train decoder only; freeze every backbone parameter", ha="center", fontsize=10, weight="bold", color=COLORS["ink"])
    save(fig, "method-overview")


def finecops_difficulty(data: dict) -> None:
    rows = [row for row in data["finecops_slices"] if row["slice"].startswith("level_")]
    rows.sort(key=lambda row: row["slice"])
    x = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharex=True)
    for variant, color in (("A4", COLORS["blue"]), ("S4", COLORS["red"]), ("A8", COLORS["green"])):
        axes[0].plot(x, [100 * float(row[variant]) for row in rows], marker="o", color=color, label=variant)
    axes[0].set(title="FineCops-Ref official difficulty", ylabel="IoU@0.5 (%)", xlabel="Level", xticks=x)
    axes[0].grid(alpha=.3, color=COLORS["grid"]); axes[0].legend(frameon=False)
    for label, color in (("delta_a4_s4", COLORS["blue"]), ("delta_a8_s4", COLORS["green"])):
        low, high = ("a4_s4_ci95_low", "a4_s4_ci95_high") if label == "delta_a4_s4" else ("a8_s4_ci95_low", "a8_s4_ci95_high")
        values = [100 * float(row[label]) for row in rows]
        axes[1].errorbar(x, values, yerr=[[value - 100 * float(row[low]) for value, row in zip(values, rows)], [100 * float(row[high]) - value for value, row in zip(values, rows)]], marker="o", capsize=3, color=color, label="A4-S4" if label == "delta_a4_s4" else "A8-S4")
    axes[1].axhline(0, color=COLORS["ink"], linewidth=.8); axes[1].set(title="Paired deltas", ylabel="IoU@0.5 (pp)", xlabel="Level", xticks=x)
    axes[1].grid(alpha=.3, color=COLORS["grid"]); axes[1].legend(frameon=False)
    fig.tight_layout(); save(fig, "finecops-difficulty")


def efficiency_figure(data: dict) -> None:
    rows = data["results"]["efficiency"]["variants"]
    labels = list(rows)
    params = [rows[label]["trainable_parameters"] / 1e6 for label in labels]
    latency = [1000 * rows[label]["decoder_only"]["mean_seconds"] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2))
    axes[0].bar(labels, params, color=[COLORS["blue"], COLORS["red"], COLORS["green"]]); axes[0].set(title="Trainable decoder size", ylabel="Parameters (M)")
    axes[1].bar(labels, latency, color=[COLORS["blue"], COLORS["red"], COLORS["green"]]); axes[1].set(title="Cached decoder latency", ylabel="ms / example")
    for axis in axes: axis.grid(axis="y", alpha=.3, color=COLORS["grid"]); axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save(fig, "efficiency-summary")


def main() -> None:
    data = build()
    (PAPER / "data").mkdir(parents=True, exist_ok=True)
    (PAPER / "data/paper-data.json").write_text(json.dumps(data, indent=2) + "\n")
    write_table(data); method_overview(); finecops_difficulty(data); efficiency_figure(data)
    print(PAPER / "data/paper-data.json")


if __name__ == "__main__":
    main()
