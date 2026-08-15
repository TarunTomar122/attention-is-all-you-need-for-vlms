# Decision: Freeze the decoder-only FFN intervention

- Date: 2026-08-12
- Status: Accepted

## Context

Removing all FFNs from a VLA action expert would conflate contextual retrieval with continuous
action synthesis. Existing attention-map grounding work also makes a generic "attention grounds"
claim non-novel.

## Decision

Compare matched trainable single-query grounding decoders over frozen VLM image and text features.
`A4` deletes only the FFN residuals, `S4` retains them, and `A8` reallocates capacity to attention
depth. Freeze the backbone, readout, supervision, box conversion, and evaluation contract.

## Verification

The architecture, controls, datasets, and evaluation contracts are recorded in
[`docs/architecture.md`](../docs/architecture.md), [`docs/controls.md`](../docs/controls.md),
[`docs/datasets.md`](../docs/datasets.md), and [`docs/evaluation.md`](../docs/evaluation.md).

## Limitation

The intervention supports a claim about a small grounding decoder only. It cannot establish that a
whole VLM or a continuous action generator can omit FFNs.
