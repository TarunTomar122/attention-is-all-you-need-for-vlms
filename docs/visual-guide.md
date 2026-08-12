# Visual guide

## The question in one picture

```mermaid
flowchart LR
    I["Image"] --> V["Frozen vision encoder"]
    E["Referring expression"] --> T["Frozen text encoder"]
    V --> P["24 × 24 image patches"]
    T --> W["Text-token features"]
    P --> AO["Attention-only decoder"]
    W --> AO
    P --> ST["Standard decoder"]
    W --> ST
    AO --> HA["Patch heatmap"]
    ST --> HS["Patch heatmap"]
    HA --> C["Same deterministic box conversion"]
    HS --> C
    C --> Q["Where does removing FFNs hurt?"]
```

The frozen encoders are identical within a comparison. Only the small grounding decoder changes.

## What one dataset example contains

```mermaid
flowchart LR
    R["One record"] --> I["Image pixels"]
    R --> E["Expression: the red cup beside the laptop"]
    R --> B["Target box: x1, y1, x2, y2"]
    R --> M["Metadata: dataset, split, IDs, image size"]
    E --> K["Predeclared expression tags"]
    B --> H["24 × 24 target heatmap"]
```

Conceptually, the normalized record looks like:

```json
{
  "id": "refcocog:12345",
  "image": "COCO_train2014_000000123456.jpg",
  "width": 640,
  "height": 480,
  "expression": "the red cup beside the laptop",
  "box_xyxy": [410.0, 220.0, 458.0, 306.0],
  "tags": ["attribute", "relation"],
  "stratum": "relational",
  "compositional": false
}
```

The tags and stratum are analysis metadata. They are never supplied to the model.

## Expression types inside the datasets

![Schematic examples of four referring-expression strata](figures/reference-types.svg)

These are primarily **types of expressions within each dataset**, not four separate datasets. The datasets emphasize different parts of the spectrum:

The exact deterministic assignment and audit procedure are in the [task taxonomy](task-taxonomy.md).

| Dataset | What its examples tend to test | How we use it |
| --- | --- | --- |
| RefCOCOg | Longer descriptions and more composition | Main study and complete depth sweep |
| RefCOCO+ | Appearance and relations without absolute location words | Tests whether attention-only works without easy left/right shortcuts |
| RefCOCO | Shorter classic references, including location cues | Legacy replication and comparison with prior work |
| Ref-Adv-s | Hard distractors, long expressions, negation and ordinal/comparative language | Test-only stress benchmark |

## Architecture comparison

Both models carry one learned grounding query. It repeatedly reads the frozen text and image tokens.

```mermaid
flowchart TB
    Q0["Learned grounding query"] --> TX["Cross-attend to text"]
    TX --> IX["Cross-attend to 576 image patches"]
    IX --> F{"Decoder variant"}
    F -->|"Attention-only"| N["Next block"]
    F -->|"Standard"| FFN["Token-wise FFN"]
    FFN --> N
    N --> R["Final shared Q/K attention readout"]
    R --> H["24 × 24 probability heatmap"]
    H --> B["Bounding box"]
```

The standard model gets an FFN after every block. The attention-only model deletes exactly those FFNs. The final readout, supervision, preprocessing and box conversion remain identical.

## What success and failure look like

```mermaid
flowchart LR
    D["Direct and attribute references"] -->|"Small A4–S4 gap"| RET["Attention is enough for retrieval"]
    R["Relations and logical operators"] -->|"Larger A4–S4 gap"| COMP["FFNs help contextual computation"]
    X["Similar gap everywhere"] --> AMB["No clean retrieval–reasoning boundary"]
    Y["A8 closes the gap"] --> CAP["Depth/capacity can replace FFNs"]
    Z["A8 still fails selectively"] --> FUNC["Evidence for an FFN-specific advantage"]
```

The experiment remains useful if the expected boundary does not appear; that outcome rejects the hypothesis rather than being hidden.

## Dataset flow and leakage boundary

```mermaid
flowchart TB
    GTR["RefCOCOg UMD train"] --> GM["Train full model matrix"]
    GVA["RefCOCOg UMD validation"] --> SEL["Select checkpoints and one global heatmap threshold"]
    GM --> GTE["RefCOCOg UMD test"]
    SEL --> GTE
    GM --> ADV["Ref-Adv-s test only"]

    PTR["RefCOCO+ UNC train"] --> PM["Train separate core matrix"]
    PVA["RefCOCO+ UNC validation"] --> PM
    PM --> PTE["RefCOCO+ testA / testB"]

    RTR["RefCOCO UNC train"] --> RM["Train separate core matrix"]
    RVA["RefCOCO UNC validation"] --> RM
    RM --> RTE["RefCOCO testA / testB"]
```

The classic datasets reuse COCO images, so their training sets are never merged. Ref-Adv-s is never used for tuning.
