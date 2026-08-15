# Paper outline

## One-sentence story

In a small grounding decoder over frozen VLM features, deleting FFNs preserves performance across standard and adversarial referring-expression grounding; the small controlled-compositional deficit at fixed depth is recovered by reallocating capacity to attention depth.

## 1. Introduction

- Grounding heads conventionally inherit attention-plus-FFN Transformer blocks, yet their input already contains image and language context from a frozen VLM.
- Ask the causal question: what changes when the trainable decoder alone loses token-wise FFNs?
- State the answer early: no broad failure boundary appears; FineCops reveals a small deficit that A8 recovers.
- Contributions: matched ablation, difficulty-oriented evaluation/controls, and evidence-derived release.

## 2. Setup

- Frozen SigLIP2 image/text encoders supply 24×24 image tokens and text states.
- A4 and S4 differ only by the four FFN residuals. A8 is parameter-matched, not compute-matched.
- One learned query alternates text and image cross-attention, then a shared patch-attention readout produces the heatmap and box.
- Freeze `tau=0.8` from validation before all held-out/OOD evaluation; use paired image-clustered bootstrap across three seeds where available.

## 3. Evaluation design

- RefCOCOg: primary three-seed paired test with modality shuffles.
- Ref-Adv-s: evaluation-only adversarial set, predefined metadata slices.
- FineCops-Ref: controlled compositional test, official difficulty/tuple labels only.
- RefCOCO seed-0 and frozen CLIP one-seed: descriptive replication evidence.
- Efficiency: decoder-only and end-to-end measurements must stay separate.

## 4. Results

- RefCOCOg: direct retention passes, no confirmed logical interaction.
- Ref-Adv-s: A4 is slightly ahead overall; no monotonic boundary across hard metadata slices.
- FineCops: A4−S4 is −0.52 pp, but A8−S4 is +0.26 pp. Level 3 is noisy and does not support a harder-means-more-FFN-needed claim.
- CLIP and shuffles: result is not just a SigLIP2 or single-modality shortcut.
- Efficiency: A4 is 44.4% smaller and 10.1% faster only in decoder time; full pipeline remains backbone-dominated.

## 5. Limits and interpretation

- Do not equate this with an attention-only VLM.
- Do not call one-seed controls confirmatory.
- A8 recovery identifies a viable reallocation, not a mechanism proving that FFNs caused the gap.

## 6. Related work and conclusion

- Separate attention-map grounding from trainable FFN ablation.
- Position the language-model attention-only study as architectural context, not direct visual evidence.
- Conclude with the bounded architectural result and release commitments.
