# Decoder efficiency

Cached-feature timings isolate the trainable decoder; full-pipeline timings include image/text preprocessing and the frozen SigLIP2 backbone. CUDA synchronization, warmups, and repeated measurements are recorded in `measurements.json`.

| Variant | Params | MACs/example | Decoder ms | Decoder ex/s | Full pipeline ms | Full ex/s | Decoder peak MB | Full peak MB | Decoder latency vs S4 | Full latency vs S4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 | 2,639,104 | 501,694,464 | 6.71 ± 0.03 | 2383.03 | 209.35 ± 21.86 | 76.43 | 806.2 | 940.9 | -10.1% | -0.4% |
| S4 | 4,743,424 | 503,791,616 | 7.46 ± 0.26 | 2143.37 | 210.20 ± 18.84 | 76.12 | 817.3 | 948.9 | +0.0% | +0.0% |
| A8 | 4,752,640 | 839,598,080 | 12.70 ± 0.03 | 1259.53 | 199.91 ± 0.54 | 80.03 | 818.3 | 948.9 | +70.2% | -4.9% |