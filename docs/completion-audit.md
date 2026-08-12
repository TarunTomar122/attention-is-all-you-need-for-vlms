# GPU-readiness completion audit

Status: ready for the first bounded CUDA smoke run

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
| Backup | Local `main` and GitHub `origin/main` are identical at the audited commit |

## Intentionally pending GPU evidence

- clean-environment installation on Ubuntu/CUDA;
- pinned backbone weight download and exact 576-patch assertion;
- CUDA forward/backward smoke run;
- actual batch-32 memory fit and measured latency;
- learning-rate pilot and all training results;
- validation selection of heatmap mass;
- in-domain, control, OOD, and replication results.

These are execution results, not unresolved design choices. The first GPU action must be the bounded smoke run in the runbook. Any required deviation is logged and pushed before continuing.

## Audit checks run on 2026-08-12

```text
ok: taxonomy, data normalization, decoder, masks, gradients, geometry, and parameter match
ok: paired image-clustered analysis fixture
```

No model weights, COCO images, Ref-Adv-s images, training run, or rented GPU resource was used.
