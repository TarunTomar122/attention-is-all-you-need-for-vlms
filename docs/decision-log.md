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

## 2026-08-12 — Carry `3e-4` into the next experiment provisionally

- **Decision:** Use learning rate `3e-4` for the next matched decoder comparison, subject to the planned research checkpoint.
- **Evidence:** In the six-run, 500-update pilot, the paired mean validation loss was `5.8088` at `3e-4`, versus `5.8785` at `1e-4` and `5.8328` at `1e-3`.
- **Scope:** This selects an efficient starting setting for the current frozen SigLIP2/RefCOCOg pipeline; it does not establish final accuracy or universal optimizer behavior.
- **Revisit when:** A longer pilot, changed batch/data pipeline, unfrozen backbone, or final held-out evaluation provides evidence that the setting should change.

## 2026-08-12 — Start with a paired full-budget seed-0 gate

- **Decision:** Run only `A4` and `S4` at seed 0 for the full 5,000-update RefCOCOg budget before expanding to other seeds and controls.
- **Why:** This preserves the primary causal comparison while limiting GPU spend until the short pilot's near-tie survives the real training budget.
- **Operational contract:** Use the shared `3e-4` rate, identical data/order settings, validation every 500 updates, separate immutable output directories, and persistent unbuffered logs. Stop for non-finite loss, OOM, manifest mismatch, or a missing modality signal.
- **Revisit when:** Both checkpoints and validation summaries are complete; then inspect the A4/S4 accuracy and resource results before authorizing the wider matrix.

## 2026-08-12 — Freeze validation box mass at `0.8`

- **Decision:** Use heatmap mass `0.8` for subsequent predictions in this study.
- **Evidence:** The predeclared `S4` RefCOCOg validation selection compared `{0.5, 0.6, 0.7, 0.8, 0.9}` and selected `0.8` with IoU@0.5 `0.5421`.
- **Scope:** This is a single validation-derived post-processing choice; it is frozen before any test evaluation and applies to both decoder variants.
- **Revisit when:** Only if the protocol changes before test access; do not retune on test or OOD data.

## 2026-08-12 — Seed-0 validation does not show an immediate FFN gap

- **Decision:** Proceed to the locked held-out evaluation rather than redrawing the architecture.
- **Evidence:** At the frozen mass `0.8`, `A4` and `S4` had nearly identical validation IoU@0.5 (`0.5425` vs `0.5421`) and mean IoU (`0.4867` vs `0.4866`).
- **Limit:** One seed and validation data cannot establish the main claim; seeds 1–2, controls, and image-clustered test analysis remain required.
- **Revisit when:** Paired test results or interpretation gates contradict this validation pattern.

## 2026-08-12 — Seed-0 held-out gates pass; continue replication

- **Decision:** Continue with paired seeds 1 and 2 on RefCOCOg before expanding to the wider dataset matrix.
- **Evidence:** On the frozen RefCOCOg test mass `0.8`, A4 reached IoU@0.5 `0.5468` and S4 `0.5404`. Text shuffling reduced A4/S4 to `0.3310/0.3277`; image shuffling reduced them to `0.0627/0.0624`. Uniform and position-prior baselines were `0.0760/0.1071`.
- **Interpretation:** Both decoders use image and text information and beat fixed priors. The A4–S4 difference is not yet a claim; three seeds and clustered analysis remain required.
- **Operational note:** Singleton category groups are preserved unchanged in shuffle controls rather than aborting the full diagnostic.

## 2026-08-13 — Recover data before spending GPU budget

- **Decision:** Treat the persistent workspace as authoritative, verify every manifest image path, and delay training until the integrity gate passes.
- **Evidence:** Pod migration exposed the JSONL manifests but only part of the extracted COCO tree. The 13 GB archive was intact; the provider migration itself reported rsync failure.
- **Action:** Move the archive off the workspace quota, extract only the 25,799 unique images referenced by the frozen train/val/test manifests, and verify zero missing paths. Then run the CUDA smoke test.
- **Scope:** This is an operational reproducibility decision; it does not change the dataset, model, learning rate, or evaluation protocol.
