# Grounding decoder specification

Status: locked before implementation

## Scope of “attention-only”

“Attention-only” means the **trainable grounding decoder contains no token-wise feed-forward network or MLP**. It still contains linear Q/K/V/O projections, input projections, normalization, residual connections, softmax, and one learned grounding query. Both frozen backbones contain pretrained FFNs, so no project claim may describe the complete VLM as attention-only.

## Inputs

For each image–expression pair, the frozen backbone returns:

- image tokens `x`: `[batch, 576, image_width]`, ordered as a 24 × 24 raster grid;
- text tokens `t`: `[batch, text_length, text_width]`;
- text padding mask: `[batch, text_length]`.

Separate learned linear projections map image and text tokens to decoder width `d = 256`. These projections are shared in design across all variants, included in trainable-parameter counts, and are not multi-layer networks.

## Decoder block

The decoder carries one learned grounding query `q` of shape `[batch, 1, 256]`. Each pre-normalized block performs:

```text
q ← q + CrossAttentionText(LayerNorm(q), LayerNorm(t))
q ← q + CrossAttentionImage(LayerNorm(q), LayerNorm(x))
q ← q + FFN(LayerNorm(q))  # standard variant only
```

Locked base configuration:

- 4 blocks;
- width 256;
- 8 attention heads;
- FFN hidden width 1024 with GELU in the standard variant;
- dropout 0.0;
- learned affine LayerNorm;
- non-affine per-head Q/K RMS normalization in both variants;
- residual connections around every retained sublayer.

There is no query self-attention because there is only one grounding query. Removing it saves computation without changing the query’s receptive field.

## Shared attention readout

After the final block, a separate image-attention readout computes one logit per image patch from the final query and projected image tokens. It has learned Q/K projections but no value or output projection because only the attention weights are required.

For each head `h`:

```text
s[h, i] = dot(RMSNorm(Wq[h] q), RMSNorm(Wk[h] x[i])) / sqrt(32)
```

The head logits are averaged, then normalized across the 576 patches:

```text
p = softmax(mean_h(s[h]))
```

`p` is the model’s only localization prediction. The readout is identical for every decoder variant.

The readout is deliberately separate from the last decoder block. If the standard variant predicted from the last block’s image-attention weights, its final FFN would occur after the prediction and could not affect it.

## Supervision

Convert each ground-truth box to a 24 × 24 target distribution. For every patch cell, calculate its intersection area with the box and normalize all 576 areas to sum to one. This retains more extent information than labeling only patch centers and always gives nonzero support to a valid positive-area box.

Train with distributional cross-entropy:

```text
loss = -sum_i target[i] * log(p[i])
```

No learned box-regression or segmentation head is permitted in the main comparison.

## Deterministic heatmap-to-box conversion

1. Sum `p` across columns and rows to obtain horizontal and vertical marginals.
2. For each marginal, select the shortest contiguous interval containing at least mass `τ`; break ties by larger contained mass, then lower start index.
3. Convert the selected patch-edge intervals back through the recorded resize/crop geometry to original-image coordinates.

Select `τ` once on the validation set from `{0.5, 0.6, 0.7, 0.8, 0.9}` using the standard depth-matched model, then freeze it for all variants, datasets, backbones, and test splits. Report heatmap metrics separately so conclusions do not depend only on this conversion.

## Invariants the implementation must test

- Both variants accept identical frozen token tensors and masks.
- Both produce a finite `[batch, 576]` distribution whose rows sum to one.
- The attention-only variant contains no `FFN`, `MLP`, or two-layer token-wise projection.
- Input projections and readout have identical shapes and initialization across paired variants.
- A standard variant’s final FFN receives gradient from the localization loss.
- Padding text tokens receive zero attention probability.
- Ground-truth patch distributions are finite, nonnegative, and sum to one.
- Heatmap-to-box output stays within the original image and has positive area.

## Deliberate exclusions

- learned bounding-box head;
- CNN or SAM refinement;
- multiple object queries;
- multi-scale image features;
- backbone fine-tuning;
- auxiliary language or box-coordinate losses.

Add these only if the minimal experiment shows that patch resolution or single-query capacity, rather than FFN removal, is the dominant failure mode.
