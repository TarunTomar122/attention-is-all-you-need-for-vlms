# Decision log

Append dated decisions here; include the evidence and what would change the decision.

## 2026-08-12 — Start with frozen-VLM referring-expression grounding

- **Decision:** Compare a standard grounding decoder with an FFN-free attention-only decoder over frozen VLM features.
- **Why:** Continuous robot action generation overloads an attention-only expert. Referring-expression grounding directly tests contextual selection over real images.
- **Initial benchmark:** RefCOCO; report accuracy at IoU >= 0.5 and break out relational expressions.
- **Revisit when:** A literature review finds an identical frozen-VLM, real-image, matched-decoder ablation.

## 2026-08-12 — Study the retrieval–reasoning boundary, not a new heatmap method

- **Decision:** Evaluate matched standard and FFN-free grounding decoders to identify which visual-language reference types attention-only decoding handles, and where it fails.
- **Why:** Direct attention-map grounding is already established. Recent attention-only tool-calling work motivates the hypothesis that attention succeeds when the answer is retrievable from supplied context.
- **Revisit when:** A prior controlled study already measures this boundary on real-image visual grounding.

## 2026-08-12 — Freeze an observable expression taxonomy before evaluation

- **Decision:** Assign fixed lexical tags and one exclusive direct, absolute, relational, or logical stratum; treat composition and length as overlays.
- **Why:** The study needs auditable failure slices without presenting inferred latent reasoning as ground truth.
- **Revisit when:** The training-only manual audit exposes a systematic rule failure; changes stop once validation or test expressions are loaded.

## 2026-08-12 — Make the main claim conjunctive and practically bounded

- **Decision:** Require direct-reference equivalence within 5 points and an at-least-5-point direct-versus-logical interaction for the main claim.
- **Why:** A global average cannot establish where attention-only decoding works; practical margins and paired image-clustered intervals make the hypothesis falsifiable.
- **Revisit when:** Only before test access, with a written statistical justification independent of observed model results.

## 2026-08-12 — Add a taxonomy abstention bucket after the training audit

- **Decision:** Only unmatched expressions of eight tokens or fewer count as direct; longer unmatched text is `unclassified` and excluded from the confirmatory interaction.
- **Why:** The frozen RefCOCOg training sample showed that long unmatched descriptions often contained relations outside a conservative lexicon.
- **Revisit when:** Never for this study after validation/test access; future work can replace lexical rules with independently annotated facets.

## 2026-08-12 — Use one minimal immutable runner

- **Decision:** Keep data normalization, model logic, training/evaluation, baselines, threshold selection, and confirmatory analysis as small explicit scripts that refuse to overwrite outputs.
- **Why:** Every required comparison is reproducible without a framework, configuration hierarchy, or experiment service; immutable run directories protect evidence.
- **Revisit when:** The smoke run demonstrates a concrete missing capability such as resumable multi-hour jobs or distributed training.

## 2026-08-12 — Extract COCO on the pod system disk

- **Decision:** Keep the checksum-verified COCO archive, model cache, repository, and run artifacts on persistent `/workspace`, but extract the current pod's 82,783 image files under `/root/attention-vlm-data`.
- **Why:** RunPod's network filesystem made small-file extraction pathologically slow; local extraction completed quickly and still leaves 6.7 GB free.
- **Revisit when:** The pod is replaced, the system disk is resized, or persistent random image access proves fast enough.
