# Experimental completion audit

Status: experimental phase frozen; publication-facing paper package prepared

## Complete and locally verified

| Requirement | Evidence |
| --- | --- |
| Defensible novelty scope | [Literature audit](literature-audit.md) distinguishes the controlled real-image FFN ablation from prior attention-map grounding and synthetic attention-only studies |
| Frozen backbone choice | [Backbone contract](backbone.md) pins SigLIP 2 and CLIP revisions, token grids, preprocessing, and claims |
| Matched architecture | [Decoder specification](architecture.md) fixes tensor shapes, FFN deletion, readout, target, and box conversion |
| Fair controls | [Control matrix](controls.md) fixes causal, depth, parameter, modality, position, and optimization controls |
| Isolated datasets | [Dataset protocol](datasets.md) fixes official splits, checksums, schemas, and OOD isolation |
| Frozen task taxonomy | [Taxonomy](task-taxonomy.md) and [200-row training audit](taxonomy-audit.csv) include an explicit abstention bucket |
| Falsifiable evidence rules | [Evaluation contract](evaluation.md) fixes margins, metrics, clustered intervals, gates, and interpretation |
| Minimal implementation | `study.py`, preparation scripts, `run.py`, `baseline.py`, `select_mass.py`, and `analyze.py` cover the complete experiment path |
| CPU verification | `python3 test_study.py` passes; a synthetic three-seed paired analysis also passes |
| Reproducible execution | [GPU runbook](gpu-runbook.md) gives exact bootstrap, data, smoke, pilot, matrix, evaluation, control, and analysis commands |
| Bootstrap contract | FineCops statistics and plots regenerated with the locked image-clustered bootstrap seed `20260812`; no model or inference reran |

## Intentionally out of scope

- RefCOCO and CLIP controls are seed-0/one-seed descriptive replications, not new three-seed confirmation claims.
- RefCOCO+ was deliberately not launched; it would add cost without a sharper test of the observed conclusion.
- FineCops level-3 and small tuple-type slices remain noisy; they do not establish a monotonic difficulty boundary.
- The study does not test backbone fine-tuning, multiple queries, segmentation, or a fully attention-only VLM.

## Completed GPU evidence

- RunPod environment verified on an RTX A5000 with PyTorch 2.4.1+cu124 and the pinned SigLIP 2 revision.
- All 82,783 COCO train2014 images and the RefCOCOg train/validation manifests were prepared and verified on the pod.
- CUDA smoke run completed with finite loss, checkpoint metadata, and 1.28 GiB peak allocated VRAM.
- Six-run learning-rate pilot completed; `3e-4` had the lowest paired mean validation loss (`5.8088`).
- RefCOCOg three-seed held-out evaluation and modality shuffles completed at frozen `tau = 0.8`.
- RefCOCO seed-0 `D0/A4/S4/A8` replication completed with immutable checkpoints and held-out outputs.
- Ref-Adv-s evaluation-only analysis completed for all three RefCOCOg-trained seeds; no Ref-Adv training or tuning occurred.
- FineCops-Ref evaluation completed for A4/S4/A8 across three seeds. The locked seed-`20260812` bootstrap gives A4−S4 `−0.52 pp` (95% CI `[−0.95, −0.12]`) and A8−S4 `+0.26 pp`.
- Decoder and full-pipeline latency were measured; the CLIP-family one-seed control completed.

These are completed execution results. No further GPU experiment is authorized by this audit.

## Publication checks run on 2026-08-15

```text
ok: taxonomy, data normalization, decoder, masks, gradients, geometry, and parameter match
ok: paired image-clustered analysis fixture
ok: paper/result artifacts regenerate from committed evidence
ok: source-only archive preflight checks the exact submission package
```

The current audit records the bounded completed GPU evidence and the publication-facing CPU-only
artifact path. No final claim remains blocked by an authorized GPU experiment.

## Frozen result record

- Primary evidence: [`results/refcocog-three-seed-summary.md`](results/refcocog-three-seed-summary.md).
- Adversarial evidence: [`results/refadv/`](results/refadv/).
- Controlled compositional evidence: [`results/finecops/`](results/finecops/).
- Efficiency and second-backbone controls: [`results/efficiency/`](results/efficiency/) and [`results/clip-control/`](results/clip-control/).
- Interpretation boundary: [`paper_findings.md`](paper_findings.md).
