# Paper findings (working synthesis)

This file is the evidence map, not a claim that exceeds the completed evaluations. The fixed comparison is a frozen SigLIP2 VLM followed by a small one-query grounding decoder: A4 uses four attention-only blocks, S4 uses four attention-plus-FFN blocks, and A8 uses eight attention-only blocks. All box extraction uses the frozen validation choice `tau = 0.8`.

## Main answer

FineCops classification: **Case B**. See the full level and tuple-type table in `docs/results/finecops/finecops_summary.md` and the paired intervals in `docs/results/finecops/finecops_bootstrap.json`.

The study should distinguish the overall paired comparison from the official level-3 difficulty boundary. A8 recovery is evidence about attention capacity; it is not, by itself, proof that the FFN caused a gap.

## Evidence by benchmark

- **RefCOCOg:** three-seed result is already versioned in `docs/results/refcocog-three-seed-summary.md`. Direct A4−S4 retention was +0.26 pp (90% CI −0.33 to +0.84); the logical interaction gate was not confirmed.
- **RefCOCO:** the frozen seed-0 held-out evaluation gives testA A4/S4/A8 IoU@0.5 of 72.65/71.22/73.11% and testB 63.95/63.75/64.81%. This is a useful replication snapshot, not a three-seed confirmatory result; the checkpoint audit and full metrics are in `docs/results/refcoco/refcoco_seed0.md`.
- **Ref-Adv-s:** A4 8.11% vs S4 7.33% IoU@0.5, A8 7.91%, with A4−S4 +0.79 pp (95% CI +0.06 to +1.52). The prepared length/distractor slices did not reveal an A4 failure boundary.
- **FineCops-Ref:** official positive-test levels and tuple types are reported without invented semantic labels. This is the primary controlled compositional difficulty test for an FFN advantage.

## Efficiency

Cached-feature decoder timings isolate the trainable head; full-pipeline timings include preprocessing and the frozen backbone.

| Variant | Trainable params | MACs/example | Decoder latency | Full-pipeline latency |
| --- | ---: | ---: | ---: | ---: |
| A4 | 2,639,104 | 501,694,464 | 6.71 ms | 209.35 ms |
| S4 | 4,743,424 | 503,791,616 | 7.46 ms | 210.20 ms |
| A8 | 4,752,640 | 839,598,080 | 12.70 ms | 199.91 ms |

Full raw measurements are in `docs/results/efficiency/measurements.json`; percentage changes relative to S4 are in the generated table.

## Backbone transfer

The CLIP-family control is deliberately one seed and matched on RefCOCOg training budget, decoder definitions, loss, learning rate, and frozen mass. Its result is descriptive transfer evidence, not a new significance claim:

# CLIP-family backbone control

This is a minimal one-seed matched RefCOCOg test because the control is intended to test backbone transfer, not add a new matrix.

| Metric | A4 | S4 | A4−S4 |
| --- | ---: | ---: | ---: |
| acc_iou_0.5 | 50.8852% | 50.7082% | +0.1770% |
| iou | 0.4872 | 0.4894 | -0.0022 |
| pointing | 81.4934% | 81.4934% | +0.0000% |
| target_mass | 0.5166 | 0.5170 | -0.0004 |

N = 9602. This result is descriptive and should not be treated as a three-seed significance test.


## Defensible conclusion

FineCops shows a small overall A4 deficit that A8 recovers. The official level-3 interval is inconclusive, so frame this as an attention-capacity result rather than a monotonic difficulty boundary. Limit the claim to this frozen-VLM grounding decoder, and retain the limitations that RefCOCO is seed-0-only and the CLIP control is one seed. Do not claim universal attention-only vision reasoning.
