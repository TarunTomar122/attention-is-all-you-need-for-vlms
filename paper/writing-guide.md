# Writing Guide

## One-Sentence Story

For a small single-query grounding decoder over frozen VLM representations, deleting FFN residuals
is broadly benign; the small controlled-compositional gap at fixed depth is recovered by moving
capacity into additional attention depth.

## Abstract Requirements

Define A4, S4, A8, frozen VLM features, and the one-box grounding output. Report only evidence that
appears in the frozen table: RefCOCOg direct retention, Ref-Adv-s A4-S4 `+0.79 pp`, FineCops A4-S4
`-0.52 pp`, A8-S4 `+0.26 pp`, and the decoder-only efficiency numbers. End by saying the scope is
the trainable decoder, not a complete attention-only VLM.

## Introduction Requirements

Motivate the question from the distinction between context retrieval/assembly and arbitrary
token-wise transformation. Explain why attention-map grounding does not answer a matched FFN
necessity question. State the expected failure-boundary hypothesis and its mixed outcome before the
contributions. Do not present the result as a universal replacement claim.

## Method Requirements

Define the frozen image/text tokens, width 256, the one learned query, alternating text/image
cross-attention, S4's GELU FFN residual, heatmap supervision, and the mass-to-box transform. State
that A8 is parameter-matched, not compute-matched. Define the selected-once `tau=0.8` and the
paired image-clustered bootstrap before any results.

## Results Requirements

Use claim-evidence-caveat order. Keep RefCOCOg, Ref-Adv-s, FineCops, one-seed controls, and
efficiency distinct. Report percentage points, not relative percentages. Say an interval is
imprecise or inconclusive if it crosses zero. Do not derive a monotonic difficulty conclusion from
FineCops level 3 or small tuple slices.

## Final Editing Pass

Search the manuscript for these terms and check every occurrence:

- **attention-only:** must identify the decoder boundary.
- **hard/difficult:** must name released metadata or official benchmark labels.
- **matches/equivalent:** must identify the metric, seeds, and practical/statistical caveat.
- **faster/smaller:** must separate decoder-only from full-pipeline measurements.
- **proves/necessary:** avoid unless the narrow causal intervention and evidence actually support it.

The human author must independently verify each cited work, number, and conclusion before public
submission.
