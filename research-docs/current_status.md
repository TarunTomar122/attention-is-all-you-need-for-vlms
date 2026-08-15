# Final Research Status

Status date: 2026-08-15

## Decision

The experimental phase is complete for the first manuscript. No additional GPU experiment is
authorized by this record. The repository preserves the completed evidence and supports a CPU-only
path to regenerate the paper figures, tables, rendered review PDF, and static research page.

## Final Claim

In a small single-query grounding decoder over frozen VLM features, FFNs are often dispensable.
Fixed-depth deletion preserves performance on standard and adversarial grounding. FineCops-Ref
shows a small controlled-compositional A4 deficit, but reallocating the parameter budget into
additional attention depth recovers it. The completed slices do not show a monotonic
harder-expression boundary.

## Terminology

- **A4:** four attention-only decoder blocks.
- **S4:** the same four blocks, each with its FFN residual.
- **A8:** eight attention-only blocks, approximately parameter-matched to S4.
- **Frozen VLM:** the image and text backbone remains fixed; it still contains its pretrained FFNs.
- **Grounding:** predict one bounding box from an image and referring expression through a 24x24
  patch heatmap and the frozen mass-to-box rule.

## Completed Evidence

| Evaluation | Evidence | Reading |
| --- | --- | --- |
| RefCOCOg UMD | Three paired seeds. A4-S4 direct = +0.26 pp, 90% CI [-0.33, +0.84]. | Direct retention passes; the planned logical interaction is not confirmed. |
| Ref-Adv-s | Three seeds, 1,142 evaluation-only rows. A4-S4 = +0.79 pp, 95% CI [+0.06, +1.52]. | No collapse on released negation, length, or distractor metadata. |
| FineCops-Ref | Three seeds, 9,605 official positive-test examples. A4-S4 = -0.52 pp, 95% CI [-0.95, -0.12]. | A fixed-depth compositional gap appears. |
| A8 capacity control | A8-S4 = +0.26 pp on FineCops. | Extra attention depth recovers the observed overall gap; it is not compute-matched. |
| Frozen CLIP control | One seed, A4-S4 = +0.18 pp. | Directional replication only. |
| Modality shuffles | Image and text shuffle both sharply lower A4 and S4. | The learned heads use both modalities. |
| Efficiency | A4 has 44.4% fewer trainable decoder parameters and 10.1% lower cached-decoder latency. | Full pipeline is backbone-dominated. |

## Evidence Boundary

- This is an FFN-free **decoder** result, not a fully attention-only VLM result.
- FineCops level-3 and small tuple-type slices are imprecise; they do not establish a monotonic
  difficulty boundary.
- RefCOCO and CLIP results are seed-0/one-seed descriptive replications.
- The study does not test backbone fine-tuning, segmentation, multiple queries, or raw end-to-end
  system deployment.
- Raw licensed images, provider-local checkpoints, and unversioned qualitative box exports are
  intentionally outside Git.

## Reproduce The Release Artifacts

```bash
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

This does not download a model or use a GPU. It regenerates the release figures, result table,
paper-data manifest, review PDF, and the static website assets from committed evidence.
