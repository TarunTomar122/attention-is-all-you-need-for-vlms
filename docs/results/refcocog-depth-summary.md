# RefCOCOg depth and capacity diagnostic

Held-out RefCOCOg UMD, frozen mass `0.8`, three seeds. These are descriptive averages, not additional preregistered significance tests.

| Variant | Decoder | Mean IoU@0.5 |
| --- | --- | ---: |
| D0 | masked-mean text readout | 24.83% |
| A4 | four attention-only blocks | 55.06% |
| S4 | four attention-plus-FFN blocks | 54.97% |
| A8 | eight attention-only blocks, parameter-matched to S4 | 55.99% |

The D0 floor is far below learned decoders, showing that the trained grounding decoder contributes substantial task computation. A4 and S4 are essentially tied on this aggregate, while reallocating capacity into A8 improves the descriptive mean by about 1.02 percentage points over S4. The locked stratum bootstrap remains the primary A4/S4 inference.
