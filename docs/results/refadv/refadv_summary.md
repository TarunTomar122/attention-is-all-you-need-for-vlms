# Ref-Adv-s failure-boundary summary

Protocol: pinned Ref-Adv-s revision, 1,142 test examples, frozen `tau = 0.8`, SigLIP2 RefCOCOg checkpoints, three seeds, and image-clustered bootstrap (10,000 replicates; seed 20260812).

## Overall metrics

| Model | IoU@0.5 | Mean IoU | Pointing | Target mass |
| --- | ---: | ---: | ---: | ---: |
| A4 | 8.11% | 0.1623 | 27.12% | 0.1787 |
| S4 | 7.33% | 0.1608 | 26.77% | 0.1751 |
| A8 | 7.91% | 0.1608 | 27.00% | 0.1775 |
| A4 − S4 IoU@0.5 | +0.79 pp | — | — | — |
| A8 − S4 IoU@0.5 | +0.58 pp | — | — | — |

## Slice results

See `refadv_slices.csv` for every slice, confidence interval, and sample count. Bins are inclusive empirical quartiles computed from Ref-Adv metadata before model-performance slicing.

| Slice | N | A4 | S4 | A8 | A4−S4 | A8−S4 | A4−S4 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 1142 | 8.11% | 7.33% | 7.91% | +0.79 pp | +0.58 pp | [+0.06, +1.52] pp |
| negation | 457 | 7.29% | 7.44% | 8.10% | -0.15 pp | +0.66 pp | [-1.24, +1.02] pp |
| non_negation | 685 | 8.66% | 7.25% | 7.79% | +1.41 pp | +0.54 pp | [+0.49, +2.34] pp |
| length_Q1 | 294 | 10.20% | 9.86% | 9.98% | +0.34 pp | +0.11 pp | [-0.91, +1.59] pp |
| length_Q2 | 281 | 6.88% | 6.76% | 7.12% | +0.12 pp | +0.36 pp | [-1.19, +1.42] pp |
| length_Q3 | 306 | 6.64% | 5.66% | 6.64% | +0.98 pp | +0.98 pp | [-0.54, +2.51] pp |
| length_Q4 | 261 | 8.81% | 7.02% | 7.92% | +1.79 pp | +0.89 pp | [+0.26, +3.45] pp |
| distractors_Q1 | 468 | 9.97% | 9.05% | 9.90% | +0.93 pp | +0.85 pp | [-0.36, +2.21] pp |
| distractors_Q2 | 230 | 9.28% | 8.99% | 9.42% | +0.29 pp | +0.43 pp | [-1.45, +2.03] pp |
| distractors_Q3 | 253 | 6.06% | 4.48% | 5.27% | +1.58 pp | +0.79 pp | [+0.26, +3.03] pp |
| distractors_Q4 | 191 | 4.89% | 4.89% | 4.71% | +0.00 pp | -0.17 pp | [-1.05, +1.05] pp |

## Interpretation gate

**Case A.** A4 remains within the practical margin on every sufficiently populated hard slice.

**Next-experiment recommendation:** Recommend FineCops-Ref as the next controlled compositional benchmark and measure decoder efficiency.

Official reasoning/facet annotations were not present in the prepared schema, so no semantic labels were invented. The native fields used here are negation, distractor count, image source, and human-authored status.
