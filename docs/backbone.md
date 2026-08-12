# Frozen backbone decision

Last checked: 2026-08-12

## Decision

Use **`google/siglip2-base-patch16-384`** as the primary frozen backbone and **`openai/clip-vit-large-patch14-336`** as the replication backbone.

Both produce a 24 × 24 spatial patch grid:

- SigLIP 2: 384 / 16 = 24
- CLIP: 336 / 14 = 24

The decoder therefore sees the same number and arrangement of image tokens in both experiments. Backbone-specific linear input projections map their hidden states to the shared decoder width; those projections are present in every decoder variant and counted as trainable parameters.

## Why SigLIP 2 is primary

- Its training recipe explicitly improves localization and dense visual features, making it a strong test of the hypothesis that the relevant information is already present in context.
- The fixed-resolution base checkpoint is public, ungated, Apache-2.0 licensed, and supported directly by Transformers.
- The official checkpoint is 1,501,968,264 bytes, small enough to freeze on the planned 32 GB GPU while training only a compact decoder.
- A 24 × 24 grid is materially less coarse for patch-to-box localization than the 14 × 14 grid of a 224px ViT-B/16 checkpoint.

Pinned model revision:

```text
google/siglip2-base-patch16-384@f775b65a79762255128c981547af89addcfe0f88
```

## Why retain CLIP as a replication

SigLIP 2's localization-oriented pretraining could make attention-only decoding unusually favorable. Replicating the direction of the FFN gap with classic contrastive CLIP tests whether the finding depends on that newer training recipe.

The official CLIP L/14@336px checkpoint also has a 24 × 24 grid and is 1,711,974,081 bytes. The vision encoder is larger, so this run follows the primary experiment rather than blocking it.

Pinned model revision:

```text
openai/clip-vit-large-patch14-336@ce19dc912ca5cd21c8a653c79e251e808ccabcd1
```

## Frozen-feature contract

For both backbones:

1. Run the complete pretrained image and text encoders in evaluation mode under `torch.no_grad()`.
2. Keep final-layer spatial image tokens in raster order; remove CLIP's class token.
3. Keep final-layer text token states and their attention mask.
4. Do not fine-tune, LoRA-tune, prompt-tune, or selectively unfreeze either encoder.
5. Do not precompute features for the first correctness run. Feature caching is allowed later only as an execution optimization and must preserve identical tensors.

The project claim always applies to the **new grounding decoder**, not to the frozen backbone, whose pretrained blocks contain FFNs.

## Image preprocessing contract

Convert to RGB and resize the complete image directly to the backbone's fixed square input using its declared interpolation and normalization. Do not crop: a grounding experiment cannot silently remove the referred object. Scale box x- and y-coordinates independently into the square and record both scale factors for exact inversion.

SigLIP 2 already declares a direct 384 × 384 resize. For the CLIP replication, disable its default center crop and resize directly to 336 × 336. This is a documented distribution shift for CLIP, shared by every CLIP decoder variant; the alternative would create missing or truncated ground-truth targets.

## Alternatives rejected

- **CLIP ViT-B/16 at 224px:** cheap, but its 14 × 14 grid imposes a strong localization ceiling and changes decoder token count relative to the main model.
- **CLIP L/14@336px as the only backbone:** clean historical baseline, but weaker dense-feature pretraining makes a failure ambiguous between decoder architecture and inadequate frozen spatial features.
- **SigLIP 2 NaFlex:** native aspect ratios are attractive, but variable patch grids add batching and geometry complexity before the central hypothesis is tested.
- **A full LVLM such as LLaVA:** substantially heavier and already close to the frozen-attention-head grounding literature; its language stack adds unnecessary confounds for this decoder ablation.

## Sources

- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786)
- [SigLIP 2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-384)
- [OpenAI CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md)
- [CLIP L/14@336px checkpoint](https://huggingface.co/openai/clip-vit-large-patch14-336)
