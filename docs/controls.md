# Matched controls and experiment matrix

Status: locked before training

## Primary causal comparison

`S4` versus `A4` is the main comparison:

| ID | Blocks | Width | Text attention | Image attention | FFN |
| --- | ---: | ---: | --- | --- | --- |
| `S4` | 4 | 256 | yes | yes | 4× GELU |
| `A4` | 4 | 256 | yes | yes | none |

They use identical inputs, projections, normalization, attention, readout, loss, optimizer, update count, batches, preprocessing, and checkpoint selection. The standard model differs only by adding the pre-normalized FFN residual in each block.

For a paired seed, initialize every shared parameter identically and use the same example order. Report paired per-example and across-seed differences rather than comparing unrelated best runs.

## Capacity reallocation control

`A8` doubles attention-only depth to eight blocks. At decoder width 256, it nearly exactly matches `S4` trainable parameters:

```text
one cross-attention = 4d² + 4d parameters
one A block         = 8d² + 16d = 528,384
one FFN + its norm  = 8d² + 7d  = 526,080

A8 blocks - S4 blocks = 4 × (528,384 - 526,080) = 9,216 parameters
```

Shared input projections and the readout are identical, so `A8` and `S4` differ by about 0.2% of total trainable parameters. The implementation must print and store exact counts before training.

`A8` is **parameter-matched, not compute-matched**. It tests whether attention depth can use the capacity otherwise allocated to FFNs.

## Compute interpretation

Because there is one grounding query, an FFN processes one token while image attention projects 576 source tokens. Ignoring inexpensive normalization and softmax operations, one standard FFN adds roughly 0.6% to a same-depth attention-only block's multiply–accumulate count. Therefore:

- `A4` and `S4` are effectively compute-matched but not parameter-matched;
- `A8` and `S4` are parameter-matched but not compute-matched;
- no result may claim a large runtime advantage from FFN deletion without measured latency evidence.

Record analytical MACs, peak allocated VRAM, and median inference latency after warm-up for every trained configuration.

## Depth and retrieval controls

Primary SigLIP 2 matrix on RefCOCOg UMD, each with seeds `{0, 1, 2}`:

| ID | Purpose |
| --- | --- |
| `D0` | Direct retrieval: masked-mean frozen text tokens, shared projections, and the attention readout; no decoder block. |
| `A1`, `A2`, `A4` | Measure how attention-only depth changes simple and relational grounding. |
| `S1`, `S2`, `S4` | Same-depth FFN controls. |
| `A8` | Parameter-matched attention-only control. |

SigLIP 2 core replication matrix on RefCOCO+ UNC and RefCOCO UNC, each with seeds `{0, 1, 2}`:

```text
D0, A4, S4, A8
```

Run the same core matrix on RefCOCOg UMD with the CLIP backbone only after the complete SigLIP 2 protocol succeeds. It tests whether the direction and task-wise pattern of the FFN gap survives a backbone with different pretraining—not whether absolute accuracy matches.

## Non-trained controls

- **Position prior:** average the training target distributions and use that fixed heatmap for every example.
- **Text shuffle:** evaluate each checkpoint after a fixed derangement of expressions within category-compatible batches.
- **Image shuffle:** evaluate after a fixed derangement of image features while keeping expressions unchanged.
- **Uniform heatmap:** assign probability `1/576` to every patch.

These controls identify dataset position bias and models that ignore either modality. They are diagnostic and do not replace `S4` versus `A4`.

## Shared optimization protocol

- Same fixed train/validation split and batch construction for every paired run.
- No image augmentation in the main study.
- Fixed update budget; no early stopping.
- Select the reported checkpoint by validation heatmap cross-entropy, with identical evaluation frequency.
- Choose one learning rate for all learned variants using the predeclared pilot grid `{1e-4, 3e-4, 1e-3}`. Run `A4` and `S4`, seed 0, for 10% of the final update budget; select the rate with the lowest mean validation loss across the two architectures. This pilot may choose optimization, never architecture or task buckets.
- Keep optimizer family, weight decay, warm-up fraction, scheduler, gradient clipping, batch size, precision, and update count identical after the pilot.
- Do not tune any hyperparameter on a test split or separately per architecture.

## Required stored metadata

Each run must store:

- git commit;
- full resolved configuration;
- model and dataset revisions;
- seed and ordered example manifest hash;
- exact trainable/frozen parameter counts;
- analytical MAC estimate;
- best checkpoint selected by validation loss;
- environment versions;
- raw per-example predictions and tags.

Any departure from this matrix is a new experiment and must be logged before its results are inspected.
