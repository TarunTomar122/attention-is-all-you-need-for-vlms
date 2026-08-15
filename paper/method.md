# Method contract

For an image-expression pair, a frozen backbone returns 576 image tokens in a 24×24 raster grid and padded text token states. Shared learned linear maps project both streams to width 256. A single learned query alternates text and image cross-attention in each decoder block:

```text
q ← q + CrossAttentionText(LayerNorm(q), LayerNorm(text))
q ← q + CrossAttentionImage(LayerNorm(q), LayerNorm(image))
q ← q + FFN(LayerNorm(q))  # S4 only
```

`A4` has four attention-only blocks; `S4` has the same four blocks plus GELU FFNs of hidden width 1024; `A8` has eight attention-only blocks. The shared readout turns the final query and image tokens into one softmax-normalized patch distribution. Training minimizes cross-entropy against the box's normalized patch-overlap distribution. There is no coordinate-regression MLP, CNN/SAM refinement, auxiliary loss, backbone adaptation, or multi-query head.

The test box is derived from the heatmap's shortest contiguous horizontal/vertical mass intervals. The mass threshold `tau=0.8` was selected once on RefCOCOg validation with S4, before test/OOD evaluation, then frozen for every variant and backbone.

For three-seed comparisons, metrics are averaged per example over seeds and image IDs are resampled together for 10,000 paired bootstrap replicates at the locked seed `20260812`. FineCops slicing uses only official level and tuple-type fields; Ref-Adv-s uses only its prepared native metadata. See [`../docs/architecture.md`](../docs/architecture.md), [`../docs/controls.md`](../docs/controls.md), and [`../docs/evaluation.md`](../docs/evaluation.md) for the full contract.
