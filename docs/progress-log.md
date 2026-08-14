# Progress log

Append one dated entry for every material setup, experiment, result, or blocker.

## 2026-08-13 — Publishable expansion queued

- Reattached the persistent RunPod workspace and verified the RTX A5000, CUDA runtime, RefCOCO/RefCOCO+ archives, COCO image coverage, and canonical manifests.
- Started three-seed RefCOCOg `D0` and `A8` runs plus the recovered `S4` seed-0 run; the six new D0/A8 jobs reached update 1,000 with finite validation losses while using about 13 GiB of 24.6 GiB VRAM.
- Queued immutable post-training RefCOCOg evaluations, modality shuffles, the locked bootstrap analysis, RefCOCO/RefCOCO+ four-variant replication matrices, and later CLIP/Ref-Adv work in persistent four-job batches.
- This entry records setup only; no new accuracy claim is made until raw predictions and summaries are verified.

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

## 2026-08-12 — Seed-0 held-out evaluation and controls completed

- Evaluated the immutable A4/S4 checkpoints on all 9,602 RefCOCOg UMD test expressions at frozen mass `0.8`.
- Correct-pair metrics were A4/S4 IoU@0.5 `0.5468/0.5404`, mean IoU `0.4919/0.4888`, and pointing `0.7994/0.7964`.
- Modality controls completed: text-shuffle `0.3310/0.3277` and image-shuffle `0.0627/0.0624` IoU@0.5 for A4/S4. Uniform and position-prior baselines were `0.0760` and `0.1071`.
- Fixed the shuffle runner to preserve singleton category groups; the focused CPU invariant test still passes. No test threshold or taxonomy was changed.
- Launched the locked full-budget A4/S4 paired replications for seeds 1 and 2 in persistent tmux sessions. The wider dataset matrix remains gated on the three-seed result.
- Resource guard: both processes share the RTX A5000 only after the pilot showed about 1.3 GiB peak allocated VRAM per run; the initial health check found no crash or memory pressure.

## 2026-08-12 — Primary pair passed the first checkpoint

- Both seed-0 runs reached update 500 and wrote atomic `best.pt` checkpoints without errors: `A4` validation loss `5.8121`, `S4` validation loss `5.8176`.
- At confirmation, combined VRAM was about 3.5 GiB of 24 GiB and GPU utilization was 32%; both processes remained active in tmux.
- The pair is safe to leave unattended until the 5,000-update summaries are available; no interpretation is made from this intermediate checkpoint.

## 2026-08-12 — Primary seed-0 pair and validation audit completed

- Completed the full 5,000-update RefCOCOg seed-0 pair. Final validation losses were `A4` `5.4772` and `S4` `5.4743`.
- Selected the single heatmap-to-box mass `0.8` from `S4` validation (`IoU@0.5 = 0.5421` at the selected mass; candidate masses `0.5/0.6/0.7/0.8/0.9` were evaluated without test access).
- At mass `0.8`, validation metrics were `A4`: IoU@0.5 `0.5425`, mean IoU `0.4867`, pointing `0.7904`; `S4`: IoU@0.5 `0.5421`, mean IoU `0.4866`, pointing `0.7864`.
- The direct/relational/logical IoU@0.5 slices for `A4` were `0.5716/0.5102/0.4726`; for `S4`, `0.5688/0.5192/0.4527`. These are validation diagnostics, not confirmatory test claims.
- Generated a held-out validation visual audit at `runs/visuals/validation-a4-s4-m08.png` showing target, A4, and S4 boxes for direct, relational, and logical examples. Test data remains untouched.

## 2026-08-12 — GPU safely stopped for the night

- Seed-1 and seed-2 A4 jobs reached and saved immutable `best.pt` checkpoints before the one-hour cutoff; their paired S4 jobs had not yet produced final checkpoints.
- The partial seed checkpoints are preserved on the pod for tomorrow, but are not treated as completed runs or used for test claims.
- Interrupted both training sessions cleanly, verified no training tmux sessions remained, and confirmed GPU utilization returned to `0%` with `1 MiB` allocated.
- No further jobs were launched after the cutoff. Tomorrow resumes from the preserved checkpoints only if run metadata and update state are verified; otherwise the affected pairs restart in fresh immutable directories.

## 2026-08-13 — Fresh RunPod recovery and persistent matrix restarted

- A new RTX A5000 pod was attached to the persistent workspace. The GitHub repo, RefCOCOg manifests, and 13 GB COCO archive were available; yesterday's checkpoints were not present on this container.
- The manifests retain the historical `/root/attention-vlm-data` image root, so the pod now maps that path to the persistent dataset directory. All 25,799 unique image paths referenced by train/val/test were verified present.
- The workspace quota could not hold both the COCO archive and extracted images. The archive was moved to the pod-local root disk and only manifest-referenced images were extracted persistently; no dataset bytes were redownloaded.
- The pinned 2.8.0 wheel was abandoned after the base image's CUDA-enabled `torch 2.4.1+cu124` was confirmed working. Project dependencies were installed in a system-site-packages venv; `test_study.py` passes.
- A persistent, sequential core matrix was launched after the smoke gate: `D0/A4/S4/A8 × seeds 0/1/2`, 5,000 updates each, shared `3e-4` learning rate, validation/checkpoints every 500 updates, and separate immutable output directories. The smoke run passed with finite validation losses (`6.2281`, `6.2261`) before the first full `D0` run.
- The first full matrix process was stopped before its first checkpoint after image reads from the network-mounted workspace proved too slow. The verified 25,799-image subset was copied to local pod storage, and the workload was narrowed to the direct research comparison `A4/S4 × seeds 0/1/2`.
- The focused local-storage pipeline passed `test_study.py`, started `A4` seed 0, and reached update 500 with validation loss `5.8121` and a checkpoint. The detached runner remains active with separate immutable directories and 500-update validation/checkpoint cadence.
- A4 seed 0 completed reproducibly at 5,000 updates with validation loss `5.4772`. The early S4 seed-0 reproduction was stopped before a checkpoint because yesterday's S4 result is already preserved; the GPU was redirected to new seed-1 and seed-2 runs.
- A4/S4 seeds 1 and 2 are now running in parallel in four isolated directories. At launch, GPU utilization was `96%` with `7.1 GiB` of `24.6 GiB` VRAM and host load `26.47` on 96 CPUs; no additional jobs were added because CPU preprocessing is the current ceiling.

## 2026-08-13 — RefCOCOg three-seed expansion completed

- Completed immutable `D0`, `A8`, and recovered `S4` seed-0 training runs at 5,000 updates; final validation losses were D0 `5.7638/5.7650/5.7633`, A8 `5.4668/5.4567/5.4575`, and S4 seed 0 `5.4743`.
- Evaluated all 12 correct RefCOCOg test predictions plus 12 paired text/image-shuffle controls at frozen mass `0.8`; all raw files and logs are retained under `runs/publishable-eval` on the persistent pod.
- The locked three-seed image-clustered bootstrap found A4 minus S4 IoU@0.5 `+0.26 pp` on direct references (90% CI `[-0.33, +0.84] pp`) and `-1.04 pp` on logical references. The direct-minus-logical interaction was `+1.30 pp`, with 95% CI `[-0.09, +2.66] pp`; direct retention passed, but the preregistered interaction gate did not.
- Modality controls are strong: averaged over all examples and seeds, correct minus image-shuffle was `+48.7 pp` for A4 and `+48.6 pp` for S4; correct minus text-shuffle was `+22.6 pp` and `+22.4 pp`. These controls support genuine image/text use, while the main FFN-free task-interaction claim remains unconfirmed on RefCOCOg.
- Started the four-job RefCOCO replication batch after the RefCOCOg gate. RefCOCO+ and the CLIP/Ref-Adv extensions remain queued behind the classic replication checkpoints.

## 2026-08-13 — RefCOCO replication is active

- The first four RefCOCO UNC seed-0 jobs (`D0`, `A4`, `S4`, `A8`) reached update 500 with finite validation losses `5.7237`, `5.6385`, `5.6435`, and `5.6364`.
- The long-running pod queues remain alive: after the 24 classic replication checkpoints they will export testA/testB predictions, then start the pinned CLIP and Ref-Adv stages where the environment permits.
- The RefCOCOg depth diagnostic is now version-controlled: D0 `24.83%`, A4 `55.06%`, S4 `54.97%`, and A8 `55.99%` mean IoU@0.5 across three seeds.

## 2026-08-13 — Failure-boundary run reprioritized

- Stopped only the expansion watchers that would have launched RefCOCO seeds 1–2, RefCOCO+, and CLIP; the four active RefCOCO seed-0 trainers were left untouched.
- Ref-Adv-s preparation completed at 1,142/1,142 rows after logging and clamping one sub-pixel boundary overflow (`row 261`); genuinely invalid boxes remain rejected.
- Started evaluation-only Ref-Adv-s predictions for RefCOCOg-trained SigLIP2 `D0/A4/S4/A8`, seeds `0/1/2`, at frozen `tau = 0.8`. No Ref-Adv training or tuning is performed.
- Added one deterministic analysis entrypoint that will write per-example metrics, metadata-defined length/distractor slices, negation slices, image-clustered bootstrap intervals, two plots, an interpretation, and a summary once all 12 prediction tensors finish.

## 2026-08-13 — Ref-Adv-s evaluation and boundary analysis completed

- Completed all 12 evaluation-only prediction jobs (`D0/A4/S4/A8 × seeds 0/1/2`) on all 1,142 prepared Ref-Adv-s rows at frozen `tau = 0.8`; no Ref-Adv training or tuning was performed.
- Overall IoU@0.5 was A4 `8.11%`, S4 `7.33%`, and A8 `7.91%`. The paired image-clustered bootstrap estimated A4−S4 `+0.79 pp` with 95% CI `[+0.06, +1.52] pp`; A8−S4 was `+0.58 pp`.
- The metadata slices did not show an A4 failure boundary: negation A4−S4 `−0.15 pp`, length Q4 `+1.79 pp`, and distractor Q4 `0.00 pp`; all four length and distractor bins were reported before interpretation.
- Classified the result as Case A: A4 still matches S4 overall and on the prepared hard slices. The recorded next recommendation is FineCops-Ref as a controlled compositional benchmark plus decoder-efficiency measurement; it was not launched automatically.
- Committed the seven publication artifacts under `docs/results/refadv/` (summary, per-example table, slices, bootstrap JSON, interpretation, and two plots). The four RefCOCO seed-0 jobs remain the only active training work; no RefCOCO seeds 1–2, RefCOCO+, or CLIP jobs were launched.

## 2026-08-13 — RefCOCO seed-0 jobs reached update 2,000

- The preserved `D0/A4/S4/A8` RefCOCO seed-0 trainers are all healthy at update `2,000/5,000` with finite validation losses `5.6355/5.4471/5.4586/5.4424` respectively.
- Atomic checkpoints were refreshed for all four jobs; no summary files exist yet because training is still in progress. No expansion or forbidden jobs were launched.

## 2026-08-14 — RefCOCO seed-0 replication completed and audited

- The preserved `D0/A4/S4/A8` RefCOCO UNC seed-0 jobs all reached update `5,000` and exited cleanly. Best validation losses were D0 `5.603853749`, A4 `5.341166992`, S4 `5.339118756`, and A8 `5.339440789`.
- All four immutable `best.pt` checkpoints and `summary.json` files were present and readable. Peak allocated VRAM was `1.23/1.37/1.41/1.79 GiB` for D0/A4/S4/A8 respectively; the GPU was idle after completion.
- Rechecked the locked Ref-Adv-s outputs: all 12 prediction tensors (`D0/A4/S4/A8 × seeds 0/1/2`) contain 1,142 rows at `tau = 0.8`; the per-example table has 1,142 rows, the slice table has 11 rows, and the bootstrap uses 10,000 replicates with seed `20260812`.
- No RefCOCO seeds 1–2, RefCOCO+, CLIP, FineCops-Ref, or other expansion jobs were launched. The tracked completion digest is [refcoco_seed0_summary.md](results/refcoco/refcoco_seed0_summary.md); raw checkpoints and tensors remain on the persistent pod and stay out of Git.

## 2026-08-14 — FineCops extraction and manifest prepared

- Extracted all 4,313 required GQA images with resumable parallel HTTP ranges after the provider throttled individual requests; existing files were preserved across retries.
- Prepared 9,605 positive FineCops examples. The manifest records 122 boxes clipped for a documented 1–2 pixel image-boundary rounding overflow; larger invalid boxes remain rejected.
- Started the frozen SigLIP2 FineCops evaluation for A4/S4/A8 across seeds 0/1/2 at `tau = 0.8`; no training or test-time tuning is performed.

## 2026-08-14 — CLIP control restarted with equivalent effective batch

- The first CLIP attempt stopped before producing a checkpoint because the pod's Transformers/Torch safety check rejected the cached PyTorch `.bin` file. It produced no result artifact.
- The control was restarted with cached safetensors weights and `batch_size=64, accumulate=1`, preserving the original effective batch of 64 and the 5,000-update budget. Completed SigLIP2 work was untouched.

## 2026-08-14 — Publishable expansion pipeline completed

- FineCops-Ref positive-test evaluation, decoder efficiency, and the minimal one-seed CLIP-family control were run from immutable existing checkpoints and frozen `tau = 0.8`; raw run artifacts remain under `runs/`.
- FineCops slices use only official level and tuple-type metadata; no LLM-derived labels or test-time tuning were added. Efficiency reports cached-decoder and full-pipeline measurements separately.
