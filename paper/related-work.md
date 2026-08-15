# Related-work map

## Attention maps are established grounding signals

Attention-map grounding is not the novelty. Kang et al. extract a small number of localization-relevant attention heads from a frozen full LVLM. F-LMM uses frozen LMM attention maps with trainable CNN and SAM refinement for referring segmentation. VLT uses attention/query mechanisms for referring segmentation. These establish that attention can expose grounding information, but do not test whether a matched trainable grounding decoder needs token-wise FFNs.

## Frozen VLM grounding

ReCLIP and related frozen-CLIP systems demonstrate that pretrained visual-language representations can support region scoring, while needing additional resolution for spatial reasoning. Our intervention is deliberately smaller: keep the frozen representation fixed, train one lightweight decoder, and delete only its FFN residuals.

## Attention-only Transformers

The 2026 controlled attention-only Transformer study compares standard and attention-only language models under depth/parameter/compute matching. It finds that reallocating capacity into attention depth can nearly close the language-modeling gap. This is strong architectural context, but it does not evaluate natural-image grounding or frozen VLM representations. Our A8 control tests the analogous capacity-reallocation question in a localization decoder.

## Positioning sentence

This work is a controlled real-image grounding-head ablation: unlike extracting attention from a full pretrained LVLM, we compare matched trainable decoders with and without FFNs over the same frozen image/text tokens, then test whether any gap depends on expression difficulty and whether attention depth recovers it.
