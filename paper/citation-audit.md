# Citation audit

Verified on 2026-08-15 against primary sources.

| Key | Role in this paper | Primary source |
| --- | --- | --- |
| `kang2025heads` | Closest frozen-LVLM attention-head grounding precedent | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html) |
| `wu2025flmm` | Frozen-LMM attention-map grounding with non-attention refinement | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_F-LMM_Grounding_Frozen_Large_Multimodal_Models_CVPR_2025_paper.html) |
| `ding2021vlt` | Attention/query referring segmentation precedent | [ICCV Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.html) |
| `ndubuaku2026attention` | Attention-only language-model architectural context | [arXiv](https://arxiv.org/abs/2607.18363) |
| `tschannen2025siglip2` | Primary frozen backbone | [arXiv](https://arxiv.org/abs/2502.14786) |
| `yu2016context` | RefCOCO, RefCOCO+, and RefCOCOg benchmark lineage | [ECCV / arXiv](https://arxiv.org/abs/1608.00272) |
| `liu2024finecops` | Controlled compositional benchmark | [arXiv](https://arxiv.org/abs/2409.14750) |
| `dong2026refadv` | Adversarial benchmark | [arXiv](https://arxiv.org/abs/2602.23898) |
| `vaswani2017attention` | Standard attention-plus-FFN Transformer block | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need) |
| `kamath2021mdetr` | End-to-end multimodal grounding architecture | [arXiv](https://arxiv.org/abs/2104.12763) |
| `liu2024groundingdino` | Conventional cross-modality grounding decoder | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php) |
| `subramanian2022reclip` | Frozen-CLIP referring-expression baseline | [ACL Anthology](https://aclanthology.org/2022.acl-long.357/) |
| `sikarwar2022groundcompose` | Attention-only synthetic grounded-composition precedent | [ACL Anthology](https://aclanthology.org/2022.emnlp-main.41/) |

The novelty statement is intentionally bounded: prior work shows that attention can support grounding, but the cited papers do not perform this work's matched trainable FFN-deletion ablation over frozen VLM image and text tokens.
