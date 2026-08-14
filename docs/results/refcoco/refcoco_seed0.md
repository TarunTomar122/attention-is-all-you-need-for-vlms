# RefCOCO seed-0 held-out evaluation

Existing RefCOCO UNC seed-0 checkpoints only; no new training or seeds. Mass `tau = 0.8`.

| Split | Decoder | N | IoU@0.5 | Mean IoU | Pointing | Target mass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| testA | D0 | 5657 | 40.23% | 0.4497 | 81.42% | 0.5000 |
| testA | A4 | 5657 | 72.65% | 0.5722 | 88.47% | 0.6311 |
| testA | S4 | 5657 | 71.22% | 0.5702 | 88.42% | 0.6282 |
| testA | A8 | 5657 | 73.11% | 0.5752 | 89.18% | 0.6356 |
| testB | D0 | 5095 | 41.79% | 0.4470 | 77.68% | 0.4866 |
| testB | A4 | 5095 | 63.95% | 0.5449 | 85.79% | 0.5939 |
| testB | S4 | 5095 | 63.75% | 0.5441 | 85.69% | 0.5929 |
| testB | A8 | 5095 | 64.81% | 0.5459 | 85.89% | 0.5986 |