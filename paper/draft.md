# Do Visual Grounding Decoders Need Feed-Forward Networks?

## Abstract

Do feed-forward networks (FFNs) in visual grounding decoders add essential computation once a pretrained vision-language model has already encoded image and language context? We test this with a small single-query decoder trained over frozen VLM features. A4 uses four text/image cross-attention blocks without token-wise FFNs; S4 adds an FFN residual to each matched block; A8 doubles attention-only depth to approximately match S4's parameter count. On RefCOCOg and Ref-Adv-s, A4 matches or slightly exceeds S4. On FineCops-Ref, a controlled compositional benchmark, A4 trails S4 by 0.52 percentage points in IoU@0.5, but A8 recovers the gap. Official FineCops difficulty levels do not show a monotonic increase in the A4-S4 gap. A4 uses 44.4% fewer trainable decoder parameters and reduces cached-decoder latency by 10.1%, while the frozen VLM leaves end-to-end latency nearly unchanged. The result is not an attention-only VLM: it shows that FFNs are often dispensable in a small grounding decoder over frozen VLM representations.

## 1. Introduction

Visual grounding maps an image and a referring expression, such as “the little girl beside the woman,” to one box. Contemporary grounding decoders normally use standard Transformer blocks: cross-attention to language and image tokens followed by a token-wise FFN. That design is sensible for a large generative model, but it leaves an empirical question open for a small task head. Once a frozen VLM has supplied spatial image features and contextual text states, is the FFN still necessary, or can attention retrieve and combine what is already in context?

This is not a claim that attention maps are new. Prior work has extracted grounding signals from attention in frozen LVLMs and used attention in referring segmentation. Nor is it a claim that the frozen VLM is attention-only. Our intervention is narrower and causal: hold the frozen image/text tokens, projections, attention, readout, loss, optimization, and heatmap-to-box conversion fixed; delete only the FFN residuals from the trainable decoder.

We compare A4, a four-block FFN-free decoder, to S4, the identical four-block standard decoder. A8 doubles attention-only depth to approximately match the standard model's trainable parameter count. This separates a fixed-depth deletion test from a capacity-reallocation test. We evaluate standard RefCOCOg grounding, a locked adversarial stress test, controlled FineCops composition, modality shuffles, a frozen CLIP control, and measured decoder efficiency.

The result is more nuanced than the original failure-boundary hypothesis. A4 retains RefCOCOg direct grounding and does not collapse on Ref-Adv-s. FineCops-Ref reveals a small overall A4 deficit, but A8 recovers it; its official difficulty levels do not establish a monotonic harder-expression boundary. Thus the evidence supports attention capacity as a viable substitute for FFN capacity in this decoder, not a blanket statement that FFNs are unnecessary for vision-language models.

Contributions:

1. A matched, FFN-deletion ablation for a real-image grounding decoder over frozen VLM tokens, with a parameter-matched attention-depth control.
2. A difficulty-oriented evaluation that combines standard, adversarial, and controlled compositional grounding with modality-shuffle diagnostics and paired image-clustered intervals.
3. A frozen, evidence-derived release package that records negative and mixed outcomes: no monotonic FFN failure boundary, a small FineCops gap, and its A8 recovery.

## 2. Setup

The primary frozen backbone is SigLIP2 Base/16 at 384px. It provides 576 image tokens in a 24×24 grid and contextual text tokens. Shared linear maps project both streams to width 256. One learned grounding query alternates cross-attention to text then image tokens. S4 adds a pre-normalized GELU FFN after each such pair; A4 omits it. A8 contains eight attention-only blocks and differs from S4 by roughly 0.2% in trainable decoder parameters.

All variants use the same patch-attention readout. It scores each image patch with the final query, applies a softmax over 576 patches, and trains against the target box's normalized patch-overlap distribution. The prediction contains no box-regression MLP, segmentation refinement, auxiliary loss, or backbone adaptation. A deterministic heatmap-to-box conversion uses the shortest patch intervals covering mass `tau=0.8`, selected on validation once and frozen before all tests.

For the primary comparisons, we average three paired model seeds per example and use a 10,000-replicate image-clustered bootstrap with seed `20260812`. Ref-Adv-s bins were defined from native metadata before inspecting differences. FineCops slices use only the benchmark's official difficulty levels and tuple types.

## 3. Evaluation design

RefCOCOg is the primary in-domain three-seed study. It includes modality-shuffle controls, which test whether a decoder exploits one modality or position priors. Ref-Adv-s is evaluation only: no Ref-Adv training, tuning, threshold selection, or taxonomy choice is allowed. FineCops-Ref is the principal controlled compositional evaluation. We report its official levels and tuple types without creating LLM-derived semantic tags. RefCOCO seed-0 and a one-seed frozen CLIP-L/14@336 control are descriptive replications, so they are never used to assert a new significance result.

## 4. Results

Table 1 summarizes the evidence. On RefCOCOg, A4-S4 is +0.26 pp for direct references (90% CI [-0.33, +0.84]). This passes the study's practical direct-retention gate, but the direct-minus-logical interaction is not confirmed. Correct pairing substantially exceeds both image and text shuffle for A4 and S4, ruling out a simple single-modality explanation.

On Ref-Adv-s, A4 is +0.79 pp over S4 (95% CI [+0.06, +1.52]). Negation, expression-length, and distractor-count slices do not show a monotonic A4 loss. This rejects the expectation that the FFN-free decoder would visibly collapse under this harder evaluation, while not proving equality across every individual slice.

FineCops-Ref supplies the only clear small fixed-depth deficit: A4-S4 is -0.52 pp (95% CI [-0.95, -0.12]) overall. The parameter-matched A8 is +0.26 pp relative to S4. The A4 deficit is similar at official levels 1 and 2; level 3 is imprecise and slightly favors A4. A one-hop tuple slice is the largest observed deficit (-1.32 pp), but tuple-type analysis is descriptive. Together, these results support capacity reallocation rather than an increasingly necessary FFN with harder labels.

The frozen CLIP control has A4-S4 = +0.18 pp on IoU@0.5, but is one seed and descriptive only. Efficiency follows the architecture: A4 has 2.64M decoder parameters versus S4's 4.74M and reduces cached-feature decoder latency from 7.46 ms to 6.71 ms. Full-pipeline latency is 209.35 ms versus 210.20 ms, as the frozen VLM dominates.

## 5. Limitations

The intervention applies to a small one-query box-grounding decoder, not a full VLM, segmentation head, or multi-object detector. The primary backbone is frozen and has pretrained FFNs. FineCops positive-test performance and the 24×24 patch resolution limit absolute localization quality. RefCOCO and CLIP are limited descriptive replications, and the A8 recovery is parameter-matched rather than compute-matched. The results therefore motivate targeted follow-up tests, rather than a general prescription to remove FFNs from vision-language systems.

## 6. Related work

Attention-head extraction from a frozen LVLM, attention-based referring segmentation, and frozen-CLIP region scoring are close application precedents. They do not perform a matched trainable grounding-head FFN ablation. A controlled attention-only Transformer language-model study finds that reallocating FFN budget into attention depth can close much of the loss gap. Our study asks the corresponding question for visual grounding with natural images and frozen VLM features.

## 7. Conclusion

For this frozen-VLM grounding decoder, a fixed-depth FFN deletion is surprisingly benign across standard and adversarial evaluations. A small FineCops deficit appears under controlled composition, yet additional attention depth recovers it and no monotonic hard-example failure boundary emerges. The useful conclusion is architectural and bounded: when the relevant visual and linguistic context is already supplied by a frozen VLM, attention-only decoder capacity can often replace FFN capacity for single-query grounding.
