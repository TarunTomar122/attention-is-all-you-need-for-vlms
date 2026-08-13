# RefCOCOg three-seed result

Protocol: frozen SigLIP 2 features, RefCOCOg UMD test, heatmap mass `0.8`, three paired seeds, and image-clustered bootstrap with seed `20260812` and 10,000 replicates.

| Comparison | Delta IoU@0.5 | Interval | Gate |
| --- | ---: | --- | --- |
| A4 − S4, direct | +0.26 pp | 90% CI −0.33 to +0.84 pp | retained |
| A4 − S4, logical | −1.04 pp | 95% CI −2.20 to +0.15 pp | descriptive |
| (direct delta) − (logical delta) | +1.30 pp | 95% CI −0.09 to +2.66 pp | not confirmed |

The direct-retention gate passed, but the predeclared task-interaction gate did not. Therefore this dataset does not support the stronger claim that the FFN-free decoder has a reliably larger logical deficit; it supports a cautious retrieval-versus-reasoning hypothesis for replication.

The modality controls behaved as expected. Correct minus image-shuffle was `+53.3 pp` for A4 and `+53.2 pp` for S4 overall. Correct minus text-shuffle was `+41.6 pp` and `+40.8 pp`, respectively. The raw per-example tensors and JSON analysis remain on the persistent pod under `runs/publishable-eval/`; this file is the version-controlled result record.
