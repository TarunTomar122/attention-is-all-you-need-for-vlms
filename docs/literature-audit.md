# Literature and novelty audit

Last checked: 2026-08-12

## Working conclusion

The publishable question is not whether attention maps can localize objects. That is established. The remaining question is whether a supervised grounding decoder can remove its FFNs without losing real-image grounding performance, and whether any loss concentrates in relational and compositional references rather than retrieval-like references.

No paper found in this pass performs that exact controlled comparison. This is a provisional negative search result, not proof that no such paper exists.

## Direct precedents

| Work | Setting | Relevant result | Why it does not answer our question |
| --- | --- | --- | --- |
| [Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html) (CVPR 2025) | Frozen LVLM, training-free grounding | A few selected text-to-image attention heads provide competitive localization. | It selects existing heads from a full LVLM whose blocks contain FFNs; it does not train matched grounding decoders with and without FFNs. |
| [F-LMM](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_F-LMM_Grounding_Frozen_Large_Multimodal_Models_CVPR_2025_paper.html) (CVPR 2025) | Frozen LMM attention maps, referring segmentation | Word–pixel attention is translated to masks by trainable CNN layers and refined with SAM. | Non-attention refinement remains part of the localization path. |
| [Vision-Language Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.html) (ICCV 2021) | Supervised referring segmentation | Reformulates segmentation as direct attention and uses multiple generated queries. | It is a complete segmentation architecture, not an FFN necessity ablation over frozen features. |
| [Attention as Grounding](https://aclanthology.org/2022.findings-acl.320/) (Findings of ACL 2022) | Analysis of multimodal Transformer attention | Cross-attention captures noun grounding and high-level relation information. | It analyzes representations rather than testing an FFN-free localization decoder. |

## Motivation and adjacent evidence

| Work | Evidence used here | Limit |
| --- | --- | --- |
| [A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363) (2026) | Attention-only language models nearly close the gap under parameter matching; remaining loss concentrates in parametric recall rather than context-grounded answers. | Language modeling, not visual grounding; recent preprint. |
| [Needle](https://cactuscompute.com/blog/needle) (2026) | Frames single-shot tool calling as retrieval-and-assembly and reports an attention-and-gating model without MLPs. | Project report rather than a peer-reviewed visual study; does not isolate visual reasoning types. |
| [When Can Transformers Ground and Compose](https://arxiv.org/abs/2210.12786) (2022) | A one-layer attention-only model solves attribute composition in synthetic RefEx; a second layer is needed for its relational variant. | Symbolic grid world with sparse, controlled embeddings rather than natural images and frozen VLM features. |
| [ReCLIP](https://aclanthology.org/2022.acl-long.357/) (ACL 2022) | Frozen CLIP supports zero-shot region scoring but is weak at spatial reasoning without an added resolver. | Proposal-based zero-shot system, not a learned matched decoder study. |

## Standard architectural control

[Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php) represents the conventional design direction: language-guided query selection and a cross-modality Transformer decoder trained for box localization. Its decoder uses the standard attention-plus-FFN pattern, making that pattern the conceptual control rather than a model we need to reproduce in full.

## Dataset evidence

- The [official Refer API](https://github.com/lichengunc/refer) supplies RefCOCO, RefCOCO+, and RefCOCOg annotations over COCO images.
- The [TensorFlow Datasets documentation](https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md) records the canonical split variants and notes that RefCOCO+ forbids location-based descriptions while RefCOCOg expressions are longer on average.
- [Ref-Adv](https://ref-adv.github.io/) (ICLR 2026) adds hard distractors and reasoning facets, including negation, specifically to suppress classic REC shortcuts.

## Claim boundary

Allowed if supported by results:

> We provide a controlled study of where FFN-free grounding decoders retain or lose performance over frozen vision-language features.

Not allowed:

- attention maps are a new grounding output;
- the complete VLM is attention-only;
- FFNs are unnecessary for vision-language models generally;
- a post-hoc linguistic bucket proves a cognitive notion of reasoning.
