# arXiv metadata draft

- **Title:** Do Visual Grounding Decoders Need Feed-Forward Networks?
- **Primary category:** cs.CV
- **Secondary categories:** cs.AI, cs.LG
- **Keywords:** visual grounding, referring expression comprehension, frozen vision-language models, Transformers, attention-only networks, architectural ablations

## Abstract (plain text)

Do feed-forward networks (FFNs) in visual grounding decoders add essential computation once a pretrained vision-language model has already encoded image and language context? We study this question with a single-query decoder trained over frozen VLM features. The matched A4 decoder uses four text/image cross-attention blocks without token-wise FFNs; S4 adds one FFN residual to each block; A8 doubles attention-only depth to approximately match S4's parameter count. On RefCOCOg and Ref-Adv-s, A4 matches or slightly exceeds S4. On FineCops-Ref, a controlled compositional benchmark, A4 trails S4 by 0.52 percentage points in IoU@0.5, but A8 recovers the gap. Official FineCops difficulty levels do not reveal a monotonic increase in the FFN gap. A4 uses 44.4% fewer trainable decoder parameters and reduces cached-decoder latency by 10.1%, although end-to-end latency remains backbone-dominated. These results show that FFNs are often dispensable in a small grounding decoder over frozen VLM representations, while motivating further tests beyond this decoder and task setting.
