"""Assemble the paper-facing result map from versioned experiment artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text() if path.exists() else "Pending."


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(".")); args = parser.parse_args()
    root = args.root
    fine = read(root / "docs/results/finecops/finecops_interpretation.md")
    clip = read(root / "docs/results/clip-control/clip_control_summary.md")
    efficiency = read(root / "docs/results/efficiency/efficiency_table.md")
    case = (re.search(r"\*\*(Case [ABC])\.", fine) or [None, "Pending"])[1]
    rows = []
    measurement = root / "docs/results/efficiency/measurements.json"
    if measurement.exists():
        payload = json.loads(measurement.read_text())
        for variant in ("A4", "S4", "A8"):
            row = payload["variants"][variant]
            rows.append(f"| {variant} | {row['trainable_parameters']:,} | {row['analytical_macs_per_example']:,} | {1000*row['decoder_only']['mean_seconds']:.2f} ms | {1000*row['full_pipeline']['mean_seconds']:.2f} ms |")
    efficiency_rows = "\n".join(rows) or "| Pending | — | — | — | — |"
    text = f"""# Paper findings (working synthesis)

This file is the evidence map, not a claim that exceeds the completed evaluations. The fixed comparison is a frozen SigLIP2 VLM followed by a small one-query grounding decoder: A4 uses four attention-only blocks, S4 uses four attention-plus-FFN blocks, and A8 uses eight attention-only blocks. All box extraction uses the frozen validation choice `tau = 0.8`.

## Main answer

FineCops classification: **{case}**. See the full level and tuple-type table in `docs/results/finecops/finecops_summary.md` and the paired intervals in `docs/results/finecops/finecops_bootstrap.json`.

The study should conclude only what the primary level-3 slice and its clustered interval support. A8 recovery is evidence about attention capacity; it is not, by itself, proof that the FFN caused a gap.

## Evidence by benchmark

- **RefCOCOg:** three-seed result is already versioned in `docs/results/refcocog-three-seed-summary.md`. Direct A4−S4 retention was +0.26 pp (90% CI −0.33 to +0.84); the logical interaction gate was not confirmed.
- **RefCOCO:** the current versioned completion is a seed-0 training/checkpoint audit in `docs/results/refcoco/refcoco_seed0_summary.md`; it is not a three-seed confirmatory test result.
- **Ref-Adv-s:** A4 8.11% vs S4 7.33% IoU@0.5, A8 7.91%, with A4−S4 +0.79 pp (95% CI +0.06 to +1.52). The prepared length/distractor slices did not reveal an A4 failure boundary.
- **FineCops-Ref:** official positive-test levels and tuple types are reported without invented semantic labels. This is the primary controlled compositional difficulty test for an FFN advantage.

## Efficiency

Cached-feature decoder timings isolate the trainable head; full-pipeline timings include preprocessing and the frozen backbone.

| Variant | Trainable params | MACs/example | Decoder latency | Full-pipeline latency |
| --- | ---: | ---: | ---: | ---: |
{efficiency_rows}

Full raw measurements are in `docs/results/efficiency/measurements.json`; percentage changes relative to S4 are in the generated table.

## Backbone transfer

The CLIP-family control is deliberately one seed and matched on RefCOCOg training budget, decoder definitions, loss, learning rate, and frozen mass. Its result is descriptive transfer evidence, not a new significance claim:

{clip}

## Defensible conclusion

Use the FineCops level-3 interval as the decision boundary. If it remains non-negative, frame the paper around FFN redundancy for this frozen-VLM grounding decoder plus measured decoder savings, with the limitations that RefCOCO is seed-0-only and the CLIP control is one seed. If it is negative, report the failure boundary and whether A8 closes it. Do not claim universal attention-only vision reasoning.
"""
    output = root / "docs/paper_findings.md"; output.write_text(text); print(output)


if __name__ == "__main__": main()
