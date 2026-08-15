# Data and artifact availability

## Versioned in this repository

- code for decoder construction, dataset preparation, evaluation, bootstrap analysis, and paper-asset generation;
- frozen evaluation contracts, manifests/checksums, decision/progress logs, and experiment summaries;
- FineCops and Ref-Adv per-example metric tables, slice tables, bootstrap inputs, and generated figures;
- the rendered manuscript, source, and citation audit.

## Deliberately excluded

- COCO, GQA, and Ref-Adv image files, which remain subject to their original licenses and provider terms;
- cached model weights and provider-local checkpoints;
- raw target/predicted box exports used for qualitative panels, which are not present in this checkout.

The reported results remain reproducible from the committed aggregate artifacts and analysis scripts. Any future qualitative panel must be regenerated from the preserved raw exports and retain the image-source attribution; it must not be inferred from aggregate metrics.

## External sources

- RefCOCO / RefCOCO+ / RefCOCOg annotations: [official Refer API](https://github.com/lichengunc/refer)
- Ref-Adv-s: [pinned dataset card](https://huggingface.co/datasets/dddraxxx/ref-adv-s)
- FineCops-Ref: [official repository](https://github.com/liujunzhuo/FineCops-Ref)
- COCO images: [official source](https://cocodataset.org/)
