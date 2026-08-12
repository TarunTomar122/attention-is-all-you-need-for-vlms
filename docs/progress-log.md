# Progress log

Append one dated entry for every material setup, experiment, result, or blocker.

## 2026-08-12 — Project initialized

- GPUHub instance verified: RTX 4080 Super, 32 GB VRAM, PyTorch 2.8.0 with CUDA 12.8.
- Persistent project directory: `/root/autodl-tmp/attention-is-all-you-need-for-vlms`.
- GitHub backup and remote write access configured; initial commit pushed.

## 2026-08-12 — Local working copy established

- Cloned the GitHub repository locally; GPU is now optional until training starts.
- Locked the study framing: measure the retrieval–reasoning boundary of an FFN-free visual grounding decoder.

## 2026-08-12 — GPU-free planning started

- Created the canonical GPU-readiness decision map.
- Completed a focused novelty audit and recorded the provisional research gap.
- No model weights, large datasets, training jobs, or rented compute used.

## 2026-08-12 — Backbone decision resolved

- Locked the primary and replication frozen backbones with matching 24 × 24 spatial grids.
- Verified public access, revisions, licenses where declared, and checkpoint sizes using repository metadata only.
- No model weights downloaded.

## 2026-08-12 — Decoder contract resolved

- Locked tensor shapes, decoder operations, attention readout, supervision, box conversion, and implementation invariants.
- Kept the output pathway identical across FFN and FFN-free variants.

## 2026-08-12 — Control matrix resolved

- Locked the primary causal, parameter-matched, depth, retrieval, and modality controls.
- Predeclared paired seeds and a shared bounded learning-rate pilot.
- Parameter and compute estimates remain implementation assertions that must be verified locally before training.

## 2026-08-12 — Datasets and splits resolved

- Locked training, validation, in-domain test, and OOD test roles.
- Inspected 156 MB of official classic annotation archives, recorded SHA-256 hashes, and removed the temporary copies.
- No COCO images or Ref-Adv image data downloaded.

## 2026-08-12 — Visual study guide and task taxonomy added

- Added diagrams for dataset records, expression types, model variants, outputs, and split isolation.
- Locked deterministic task strata, compositional and length overlays, and a training-only manual audit protocol.
- Passed ten dependency-free taxonomy boundary cases; no dataset or model download required.

## 2026-08-12 — Evaluation contract resolved

- Locked the confirmatory model pair, practical margins, metrics, clustered paired bootstrap, interpretation gates, and test-access boundary.
- Predeclared how the parameter-matched `A8` control changes the conclusion.

## 2026-08-12 — Core tensor path validated on CPU

- Implemented FFN-free and standard one-query decoders, paired initialization, area-weighted targets, deterministic box conversion, metrics, and the frozen lexical taxonomy.
- Passed one focused CPU check covering distributions, padding masks, final-FFN gradients, FFN absence, target geometry, box bounds, metrics, and the `A8`/`S4` parameter match.
- No model or dataset download used.

## 2026-08-12 — Training-only taxonomy audit completed

- Checksum-verified the RefCOCOg annotation archive, sampled 40 training expressions per final stratum, and published the 200-row audit.
- Removed broad relation and counting triggers, tightened position cues, and added an honest `unclassified` bucket for long unmatched descriptions.
- Deleted the temporary annotation archive after the audit; no images, validation expressions, or test expressions were loaded.
