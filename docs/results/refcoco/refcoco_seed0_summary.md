# RefCOCO seed-0 completion summary

Protocol: RefCOCO UNC, frozen SigLIP2 features, seed `0`, `5,000` updates, learning rate `3e-4`, batch size `32`, gradient accumulation `2`, warmup `250`, validation/checkpoint cadence `500`, and immutable output directories under `runs/replications/refcoco/` on the persistent pod.

## Verified checkpoints

| Decoder | Final update | Best validation loss | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| D0 | 5,000 | 5.603853749 | 1.23 GiB |
| A4 | 5,000 | 5.341166992 | 1.37 GiB |
| S4 | 5,000 | 5.339118756 | 1.41 GiB |
| A8 | 5,000 | 5.339440789 | 1.79 GiB |

All four `best.pt` and `summary.json` files were present and readable after the trainers exited. The final training logs recorded update `5,000` for every variant. No RefCOCO seeds `1–2`, RefCOCO+, CLIP, FineCops-Ref, or other expansion jobs were running at audit time; the GPU was idle.

## Relationship to the main study

The Ref-Adv-s evaluation was already complete before this replication finished: all `D0/A4/S4/A8 × seeds 0/1/2` prediction tensors contain `1,142` examples at frozen mass `tau = 0.8`. The publication artifacts and failure-boundary analysis remain in [the Ref-Adv-s result directory](../refadv/), with `10,000` image-clustered bootstrap replicates and seed `20260812`.

Raw checkpoints and prediction tensors remain on the persistent pod and are intentionally excluded from Git by the repository ignore rules.
