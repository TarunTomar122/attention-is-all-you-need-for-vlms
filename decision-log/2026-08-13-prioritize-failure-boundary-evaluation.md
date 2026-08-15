# Decision: Prioritize adversarial failure-boundary evaluation

- Date: 2026-08-13
- Status: Accepted

## Context

The initial RefCOCOg comparison did not establish the planned direct-versus-logical interaction.
Launching more near-duplicate classic-dataset seeds would spend GPU budget without sharply testing
where A4 might fail.

## Decision

Preserve completed and active RefCOCO work, freeze `tau=0.8`, and evaluate all completed
RefCOCOg-trained checkpoints on the 1,142-example Ref-Adv-s set without training or tuning. Define
continuous-variable bins from released metadata before performance slicing.

## Verification

[`docs/results/refadv/refadv_summary.md`](../docs/results/refadv/refadv_summary.md) records all
12 prediction jobs, per-example outputs, released metadata slices, and the 10,000-replicate paired
bootstrap.

## Limitation

Ref-Adv-s does not expose official reasoning facets in the prepared schema. No semantic labels were
invented.
