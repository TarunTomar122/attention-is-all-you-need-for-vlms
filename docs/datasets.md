# Dataset protocol

Status: locked before training

## Dataset roles

| Dataset | Official split | Role |
| --- | --- | --- |
| RefCOCOg | UMD | Primary study. Run the full depth and FFN-control matrix because expressions are longer and the split has image-disjoint train/validation/test partitions. |
| RefCOCO+ | UNC | Attribute-focused replication. Run `D0`, `A4`, `S4`, and `A8`; its collection protocol excludes location-based descriptions. |
| RefCOCO | UNC | Legacy REC replication. Run `D0`, `A4`, `S4`, and `A8`; report testA and testB separately. |
| Ref-Adv-s | Public release revision pinned below | Locked out-of-distribution test for the RefCOCOg-trained models. Never train, tune, select checkpoints, choose thresholds, or design taxonomy rules on it. |

Train a separate model on each classic dataset. Do not union their training sets: the datasets reuse COCO images with independently defined splits, so union training can leak images into another dataset's test partition.

The CLIP-backbone replication runs only the RefCOCOg core matrix (`D0`, `A4`, `S4`, `A8`).

## Unit of observation

Each referring expression is one example, paired with the bounding box of its referenced object. Multiple expressions may point to the same object; they remain separate examples but always follow the object/image's official split.

Counts verified directly from the official annotation archives:

| Dataset | Train expressions | Validation expressions | Test expressions |
| --- | ---: | ---: | ---: |
| RefCOCO UNC | 120,624 | 10,834 | testA 5,657; testB 5,095 |
| RefCOCO+ UNC | 120,191 | 10,758 | testA 5,726; testB 4,889 |
| RefCOCOg UMD | 80,512 | 4,896 | test 9,602 |
| Ref-Adv-s | — | — | 1,142 |

In RefCOCO and RefCOCO+, testA contains people and testB contains non-people. Always report them separately and combined only as an expression-count-weighted supplementary number.

## Canonical classic annotations

Use the [official Refer API](https://github.com/lichengunc/refer) pinned at:

```text
lichengunc/refer@e3bbaa30d2ca41cf0e5c0d3819d7e4ed9fd38fff
```

The original host is unavailable, so use the verified archived snapshots below:

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `refcoco.zip` | 53,993,993 | `7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0` |
| `refcoco+.zip` | 45,613,210 | `5f1238112d63199e68da54a28f201471909b21ded7ed79a57d51b4c1443c6b45` |
| `refcocog.zip` | 56,712,951 | `3d1f7e5b2ff2205940bf59de55f861f5f2cc1403fb980669933a7f9af1aa8211` |

Required files:

```text
refcoco/instances.json
refcoco/refs(unc).p
refcoco+/instances.json
refcoco+/refs(unc).p
refcocog/instances.json
refcocog/refs(umd).p
```

Classic images come from COCO `train2014`. Download them from the [official COCO source](https://cocodataset.org/), follow its terms, and never commit or redistribute images through this repository.

## Classic annotation contract

- `refs(...).p` is a list of reference records linking `ann_id`, `image_id`, split, and one or more sentence records.
- Sentence text comes from `sentence["sent"]`; retain `sent_id` as the stable expression identifier.
- Join `ann_id` to `instances.json`.
- COCO boxes are absolute `[x, y, width, height]`; convert exactly once to `[x1, y1, x2, y2]` in `float32`.
- Image width/height and file name come from `instances.json`.
- Reject duplicate `(dataset, split, sent_id)` identifiers.
- Fail preparation on missing images, non-finite coordinates, non-positive boxes, or boxes fully outside the image. Do not silently discard records.
- Clamp only sub-pixel overflow caused by annotation precision, and log every clamped identifier.

## Ref-Adv-s contract

Use [`dddraxxx/ref-adv-s`](https://huggingface.co/datasets/dddraxxx/ref-adv-s) pinned at:

```text
e7a53e352b5885b8228fc6afa8645ab78e76d5f1
```

The public Parquet file is 255,481,432 bytes and includes the images. Its sole Hugging Face split is named `train` for packaging; alias all 1,142 rows to `test` inside this project.

- Expression: `normal_caption`
- Box: `solution`, absolute `[x1, y1, x2, y2]`
- Image size: `width`, `height`
- Native diagnostic fields: `distractors`, `use_negation`, `image_source`, and `human_authored`
- Stable identifier: `refadv:{row_idx}`

The released subset mixes COCO val2017 and OpenImages sources and is CC BY 4.0, with image copyrights retained by their original sources. Preserve source attribution in exported qualitative examples.

## Evaluation isolation

- All optimization choices use only the current classic dataset's training and validation partitions.
- The global heatmap-to-box mass `τ` is selected once using SigLIP 2 `S4` on RefCOCOg validation, then frozen everywhere.
- Test annotations are read only during final evaluation.
- Ref-Adv-s is loaded only after all model, taxonomy, and metric decisions are committed.
- Raw predictions are stored before aggregate metrics are calculated.

## Local verification performed

Without downloading any images or retaining dataset files, the three classic annotation archives were checksum-verified and inspected for:

- required split pickle names;
- reference and expression counts;
- reference/sentence keys;
- COCO image and annotation joins;
- `[x, y, width, height]` box storage.

Temporary archive copies were removed after inspection.

## Sources

- [Official Refer API](https://github.com/lichengunc/refer)
- [TensorFlow Datasets RefCOCO documentation](https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md)
- [Ref-Adv project](https://ref-adv.github.io/)
- [Ref-Adv-s dataset card](https://huggingface.co/datasets/dddraxxx/ref-adv-s)
- [COCO](https://cocodataset.org/)
