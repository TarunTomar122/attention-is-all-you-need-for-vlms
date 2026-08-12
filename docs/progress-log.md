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
