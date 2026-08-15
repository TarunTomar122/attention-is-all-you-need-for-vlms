# Claims and limitations

## Approved headline claim

> In a small trainable grounding decoder over frozen VLM features, FFNs are often dispensable: fixed-depth removal preserves performance on standard and adversarial grounding, while a small controlled-compositional deficit is recovered by additional attention depth.

## Evidence map

| Claim | Evidence | Required caveat |
| --- | --- | --- |
| A4 retains RefCOCOg direct grounding | Three paired seeds; `+0.26 pp`, 90% CI `[-0.33, +0.84]` | The predeclared logical interaction is not confirmed. |
| A4 does not collapse on Ref-Adv-s | Three seeds; `+0.79 pp`, 95% CI `[+0.06, +1.52]` | This is an evaluation-only, prepared subset; no invented semantic slices. |
| FineCops reveals a small fixed-depth gap | Three seeds; `−0.52 pp`, 95% CI `[−0.95, −0.12]` | Official level-3 and several tuple slices are noisy; no monotonic boundary. |
| Extra attention can recover the observed overall FineCops gap | A8 is `+0.26 pp` versus S4 | A8 is parameter-matched, not compute-matched; recovery is not a proof of FFN mechanism. |
| A4 has a decoder efficiency advantage | 44.4% fewer trainable parameters; 10.1% lower decoder latency | End-to-end latency differs by only 0.4% because the frozen VLM dominates. |
| Direction transfers beyond SigLIP2 | One-seed CLIP A4−S4 = `+0.18 pp` | Descriptive only; no multi-seed interval. |

## Prohibited wording

- “attention-only VLM” or “attention is all you need for VLMs”;
- “FFNs are unnecessary for vision-language models generally”;
- “FineCops proves a monotonic reasoning boundary”;
- “A8 proves FFNs do nothing”;
- “A4 is faster end to end.”

## Material limitations

- The backbone is frozen and contributes most full-pipeline latency.
- The decoder predicts a 24×24 heatmap and one box; this is neither pixel segmentation nor multi-object detection.
- RefCOCO and CLIP replication evidence is seed-0/one-seed, respectively.
- FineCops evaluates its official positive test set and official metadata, so it should not be generalized to all compositional language.
- No fine-tuning, segmentation head, multi-query decoder, or fully attention-only backbone was evaluated.
