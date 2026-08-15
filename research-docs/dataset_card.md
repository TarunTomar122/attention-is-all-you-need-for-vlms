# Grounding Evaluation Dataset Card

## Purpose

This study evaluates a causal decoder intervention, not a new general-purpose grounding
leaderboard. Each example is an image, a referring expression, and one target bounding box. The
reported unit is a fixed prediction/evaluation pair under a frozen feature extractor.

## Completed Evaluations

| Dataset | Role | Split and status | Use in this study |
| --- | --- | --- | --- |
| RefCOCOg UMD | Primary | Official test, three paired seeds | Primary fixed-contract comparison and modality controls. |
| RefCOCO UNC | Replication | Seed-0 descriptive batch | Checks that the decoder behavior is not unique to RefCOCOg. |
| Ref-Adv-s | Adversarial/OOD | Prepared 1,142-example test-only subset | Evaluation only; no model, threshold, or slice tuning. |
| FineCops-Ref | Controlled compositional | Official positive test, 9,605 rows | Tests official level and tuple-type metadata with frozen checkpoints. |

## Determinism And Isolation

- The heatmap mass threshold `tau=0.8` was chosen once on RefCOCOg validation and frozen before
  test and OOD evaluation.
- The three-seed analyses average paired seeds per example, then resample image IDs with
  replacement for 10,000 image-clustered bootstrap replicates at seed `20260812`.
- Ref-Adv-s receives no training or selection. Its expression-length and distractor bins are
  empirical metadata quartiles defined before looking at decoder-performance differences.
- FineCops uses only released official level and tuple-type fields. No LLM-derived semantic labels
  are introduced.

## Storage And Redistribution

Third-party images, backbone weights, and provider-local checkpoints are excluded from Git. The
repository releases preparation/evaluation code, frozen contracts, aggregate results, per-example
metrics where permitted, bootstrap outputs, and hashes. Users must follow the upstream terms for
COCO, GQA, RefCOCO-family annotations, Ref-Adv-s, and FineCops-Ref.

## Known Limitations

- Bounding-box IoU cannot establish pixel-level segmentation quality.
- The patch heatmap has 24x24 spatial resolution before deterministic box conversion.
- Referring-expression benchmarks may contain priors and training-data overlap with pretrained
  models.
- Dataset names should not be interpreted as exhaustive labels for reasoning difficulty.
- A model can be near-matched overall while differing in an unreported or low-count slice.
