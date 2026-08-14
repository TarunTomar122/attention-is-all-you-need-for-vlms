# FineCops-Ref positive-test summary

Protocol: official positive test split, frozen SigLIP2 RefCOCOg checkpoints, three decoder seeds, tau = 0.8, and image-clustered bootstrap (10,000 replicates; seed 20260814).

## Overall metrics

| Model | IoU@0.5 | Mean IoU | Pointing | Target mass |
| --- | ---: | ---: | ---: | ---: |
| A4 | 27.01% | 0.2971 | 52.20% | 0.3609 |
| S4 | 27.54% | 0.2978 | 51.79% | 0.3597 |
| A8 | 27.79% | 0.2977 | 52.04% | 0.3621 |
| A4 − S4 IoU@0.5 | -0.52 pp | — | — | — |
| A8 − S4 IoU@0.5 | +0.26 pp | — | — | — |

## Official slices

| Slice | N | A4 | S4 | A8 | A4−S4 | A8−S4 | A4−S4 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 9605 | 27.01% | 27.54% | 27.79% | -0.52 pp | +0.26 pp | [-0.94, -0.12] pp |
| level_1 | 5730 | 31.47% | 32.04% | 32.06% | -0.58 pp | +0.02 pp | [-1.08, -0.05] pp |
| level_2 | 3404 | 20.82% | 21.35% | 22.04% | -0.53 pp | +0.70 pp | [-1.46, -0.10] pp |
| level_3 | 471 | 17.62% | 17.48% | 17.48% | +0.14 pp | +0.00 pp | [-1.06, +2.55] pp |
| tuple_type_0_hop | 2333 | 31.22% | 31.79% | 32.30% | -0.57 pp | +0.51 pp | [-1.32, +0.00] pp |
| tuple_type_1_hop | 2146 | 26.72% | 28.04% | 27.80% | -1.32 pp | -0.23 pp | [-2.06, -0.49] pp |
| tuple_type_2_hop | 2555 | 21.97% | 22.05% | 23.04% | -0.08 pp | +0.99 pp | [-0.66, +0.86] pp |
| tuple_type_and | 1639 | 28.72% | 28.82% | 29.06% | -0.10 pp | +0.24 pp | [-1.05, +0.89] pp |
| tuple_type_same_attr | 705 | 30.83% | 31.44% | 29.50% | -0.61 pp | -1.94 pp | [-1.83, +0.89] pp |
| tuple_type_same_attr_two_hop | 227 | 19.24% | 19.53% | 20.41% | -0.29 pp | +0.88 pp | [-2.31, +2.64] pp |

## Classification: Case B

A4 has a confidence interval below zero overall and A8 closes the observed gap; the level-3 slice itself is not a confirmed monotonic boundary.

See `finecops_interpretation.md` for the restrained conclusion and next-experiment recommendation.
