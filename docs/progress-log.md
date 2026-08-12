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

## 2026-08-12 — GPU execution path prepared

- Added deterministic classic and Ref-Adv-s preparation, fixed-budget training, immutable checkpoint selection, raw prediction export, modality shuffles, fixed priors, mass selection, and paired image-clustered analysis.
- Added exact environment, data, smoke-run, pilot, matrix, evaluation, and stop instructions in the GPU runbook.
- Recorded analytical trainable MACs, peak allocated VRAM, and measured median inference-batch latency in run artifacts.
- GPU execution remains intentionally pending; no model weights, COCO images, Ref-Adv-s images, or training runs were started locally.

## 2026-08-12 — GPU-free goal completed

- Re-ran the focused CPU invariant check and a synthetic three-seed paired image-clustered analysis fixture successfully.
- Audited every readiness decision as resolved and separated locally verified work from GPU-only pending evidence.
- Verified local `main` and GitHub `origin/main` matched before the final audit commit.

## 2026-08-12 — RunPod GPU smoke passed

- Cloned commit `9f7017a` on an RTX A5000 (24 GB), using PyTorch 2.4.1+cu124 and the pinned SigLIP 2 revision.
- Verified all 82,783 COCO train2014 images and generated canonical RefCOCOg manifests: 80,512 train, 4,896 validation, and 9,602 test examples.
- Ran the real `A4` attention-only decoder for two optimizer updates at effective batch size 64; validation loss decreased from 6.2281 to 6.2261.
- Saved immutable metadata, summary, and checkpoint artifacts; peak allocated VRAM was 1.28 GiB and trainable parameters were 2,639,104.

## 2026-08-12 — RunPod learning-rate pilot completed

- Completed all six fixed-budget runs on the RTX A5000: `A4` and `S4` at `1e-4`, `3e-4`, and `1e-3`, each for 500 updates with seed 0.
- Best validation losses were: `A4/S4 @ 1e-4` = `5.8823/5.8747`; `A4/S4 @ 3e-4` = `5.8074/5.8102`; `A4/S4 @ 1e-3` = `5.8103/5.8552`.
- The mean across variants was lowest at `3e-4` (`5.8088`); all six immutable checkpoints and summaries were verified on the pod.
- No OOM, crash, or stalled process occurred. Peak allocated VRAM remained about 1.3 GiB for `A4` and 1.3 GiB for `S4`; the GPU was adequate for this frozen-backbone pilot.
- This is a provisional optimizer selection from a short loss-only pilot, not final grounding accuracy or a benchmark result.

## 2026-08-12 — Primary seed-0 pair launched

- Started the first full-budget paired comparison on RunPod: SigLIP 2, RefCOCOg UMD, seed 0, `A4` versus `S4`, learning rate `3e-4`, 5,000 updates, global batch 64.
- Runs are isolated in separate tmux windows and immutable output directories: `runs/refcocog-siglip2-A4-s0` and `runs/refcocog-siglip2-S4-s0`.
- Validation runs every 500 updates; logs are streamed with unbuffered output and best checkpoints are written atomically. The pair is the first go/no-go check before seeds 1 and 2 or the wider matrix.
- Resource guard: both processes share the RTX A5000 only after the pilot showed about 1.3 GiB peak allocated VRAM per run; the initial health check found no crash or memory pressure.
