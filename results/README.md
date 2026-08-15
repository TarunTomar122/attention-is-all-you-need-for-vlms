# Results Index

This directory is the release-facing map of the frozen evidence. The detailed aggregate artifacts,
per-example metric tables, bootstrap outputs, and generated diagnostic plots remain under
[`docs/results/`](../docs/results/) so that the static research page can ship them together.

## Main Evidence

| Result family | Canonical artifact | Scope |
| --- | --- | --- |
| RefCOCOg | [`docs/results/refcocog-three-seed-summary.md`](../docs/results/refcocog-three-seed-summary.md) | Three-seed primary comparison and modality controls. |
| RefCOCO | [`docs/results/refcoco/`](../docs/results/refcoco/) | Seed-0 descriptive classic-dataset replication. |
| Ref-Adv-s | [`docs/results/refadv/`](../docs/results/refadv/) | Three-seed evaluation-only adversarial/failure-boundary analysis. |
| FineCops-Ref | [`docs/results/finecops/`](../docs/results/finecops/) | Three-seed controlled compositional evaluation. |
| Efficiency | [`docs/results/efficiency/`](../docs/results/efficiency/) | Cached-decoder and full-pipeline measurements. |
| CLIP control | [`docs/results/clip-control/`](../docs/results/clip-control/) | One-seed frozen-backbone directional replication. |

## Result Reading Order

1. Read the [current status](../research-docs/current_status.md) for the approved conclusion.
2. Inspect `refcocog-three-seed-summary.md` for the primary gate and shuffles.
3. Inspect Ref-Adv-s before claiming any broad hard-example collapse.
4. Inspect FineCops-Ref and its paired bootstrap before discussing the fixed-depth gap or A8.
5. Read the [claims and limitations](../paper/claims-and-limitations.md) before quoting a number.

## Evidence Boundary

The committed outputs are experiment records, not licenses to retune. No future analysis should
change `tau`, bootstrap seed, checkpoint selection, or post-process behavior using Ref-Adv-s or
FineCops results. Raw images and provider-local checkpoints are intentionally excluded.
