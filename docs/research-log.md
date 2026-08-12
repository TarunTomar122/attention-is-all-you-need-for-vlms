# Research log

Append sources, claims, and limits. Do not turn hypotheses into conclusions.

## 2026-08-12 — Initial map

- **Grounding DINO (2023):** real-image grounding decoder combines self-attention, image/text cross-attention, and FFNs; it is the architectural control.
- **MDETR (2021):** end-to-end text-conditioned detection with full Transformer fusion.
- **ReCLIP (2022):** frozen CLIP can score candidate regions zero-shot but has weak spatial reasoning without extra resolution logic.
- **When Can Transformers Ground and Compose (2022):** attention-only grounding succeeds on synthetic compositional scenes, leaving real-image frozen-VLM grounding as the useful test.

## 2026-08-12 — Attention-only retrieval hypothesis and direct precedents

- **Needle (2026):** a 26M attention-only model for single-shot tool calling; frames the task as retrieval-and-assembly from supplied context.
- **A Controlled Study of Attention-Only Transformers (2026):** matched attention-only models approach standard models when answers are context-grounded, while the remaining deficit concentrates in parametric recall.
- **Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding (CVPR 2025):** three frozen LVLM attention heads yield competitive training-free grounding. This rules out novelty claims based only on extracting an attention heatmap.
- **F-LMM (CVPR 2025):** frozen LMM word-pixel attention maps are useful grounding priors, but a CNN and SAM refiner turn them into masks.

Implication: test the unmeasured visual boundary—retrieval-style references versus relational, counting, and compositional references—using a controlled FFN ablation.

## 2026-08-12 — Focused novelty audit

- Confirmed the closest direct precedent is **Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding** (CVPR 2025), which selects localization heads inside frozen full LVLMs rather than training a matched FFN-free grounding decoder.
- Confirmed **F-LMM** retains trainable CNN mask decoding and SAM refinement after frozen attention maps.
- Confirmed synthetic **RefEx** results are unusually relevant: one attention-only layer handles attribute composition, while its relational variant needs two layers. Natural images and frozen VLM features remain untested in that controlled setting.
- Added Ref-Adv as a candidate shortcut-resistant evaluation set because it includes hard distractors and annotated reasoning facets.
- Full source matrix and claim limits: [literature-audit.md](literature-audit.md).

## 2026-08-12 — Backbone selected

- Selected SigLIP 2 B/16 at 384px as the primary frozen backbone because its released training recipe explicitly improves localization and dense features.
- Retained OpenAI CLIP L/14 at 336px as a replication to test whether the FFN gap depends on SigLIP 2's localization-oriented pretraining.
- Both yield 24 × 24 patch grids, avoiding a spatial-resolution confound in the decoder comparison.
- Pinned both Hugging Face model revisions and verified checkpoint sizes without downloading weights. Full rationale: [backbone.md](backbone.md).

## 2026-08-12 — Decoder definition locked

- Defined attention-only narrowly at the trainable grounding-decoder boundary; normalization, residuals, and linear attention projections remain allowed.
- Both variants carry one grounding query through repeated text and image cross-attention; only the standard variant includes token-wise FFNs.
- Added an independent final attention readout so the final standard FFN causally influences the heatmap.
- Locked area-overlap patch supervision and a shared deterministic heatmap-to-box conversion. Full specification: [architecture.md](architecture.md).
