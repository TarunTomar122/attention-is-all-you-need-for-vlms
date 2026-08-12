"""One dependency-free CPU check for the locked study invariants."""

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from prepare_classic import prepare
from run import build_decoder

from study import (
    CrossAttention, FFN, GroundingDecoder, box_target, classify_expression,
    copy_shared_parameters, heatmap_box, per_example_metrics,
)


def check() -> None:
    cases = {
        "the dog": ("direct", (), False),
        "the red cup": ("direct", ("attribute",), False),
        "person in the background": ("absolute", ("absolute",), False),
        "the cup next to the laptop": ("relational", ("relation",), False),
        "the woman wearing red beside the child": (
            "relational", ("attribute", "relation"), True,
        ),
        "the second person from the right": ("logical", ("ordinal",), False),
        "the smaller pot in front of the pan": (
            "logical", ("comparison", "relation"), True,
        ),
        "the person not holding anything": (
            "logical", ("negation", "relation"), True,
        ),
        "the elephant ridden by three people": (
            "logical", ("cardinality", "relation"), True,
        ),
        "2 dogs on the left": ("logical", ("absolute", "cardinality"), True),
    }
    for expression, expected in cases.items():
        result = classify_expression(expression)
        assert (result.stratum, result.tags, result.compositional) == expected

    torch.manual_seed(0)
    attention_only = GroundingDecoder(48, 24, width=32, heads=4, depth=2)
    standard = GroundingDecoder(48, 24, width=32, heads=4, depth=2, with_ffn=True)
    copy_shared_parameters(attention_only, standard)
    assert not any(isinstance(module, FFN) for module in attention_only.modules())

    shared = attention_only.state_dict()
    for name, value in standard.state_dict().items():
        if name in shared and shared[name].shape == value.shape:
            assert torch.equal(shared[name], value), name

    image = torch.randn(2, 16, 48)
    text = torch.randn(2, 5, 24)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    for model in (attention_only, standard, GroundingDecoder(48, 24, width=32, heads=4, depth=0)):
        output = model(image, text, mask)
        assert output.shape == (2, 16)
        assert torch.isfinite(output).all()
        assert torch.allclose(output.sum(1), torch.ones(2), atol=1e-6)

    loss = -standard(image, text, mask)[:, 0].log().mean()
    loss.backward()
    final_ffn = standard.blocks[-1].ffn
    assert final_ffn is not None
    assert all(parameter.grad is not None for parameter in final_ffn.parameters())

    attention = CrossAttention(32, 4)
    _, weights = attention(
        torch.randn(2, 1, 32), torch.randn(2, 5, 32), mask,
        return_weights=True,
    )
    for row in range(len(mask)):
        assert weights[row, ..., ~mask[row]].eq(0).all()

    boxes = torch.tensor([[20.0, 10.0, 60.0, 50.0], [0.0, 0.0, 100.0, 80.0]])
    sizes = torch.tensor([[100.0, 80.0], [100.0, 80.0]])
    targets = box_target(boxes, sizes, grid=4)
    assert targets.shape == (2, 16)
    assert torch.allclose(targets.sum(1), torch.ones(2))
    predicted_boxes = heatmap_box(targets, sizes, mass=0.5, grid=4)
    assert (predicted_boxes[:, 2:] > predicted_boxes[:, :2]).all()
    assert (predicted_boxes >= 0).all() and (predicted_boxes <= 100).all()
    metrics = per_example_metrics(targets, predicted_boxes, boxes, sizes, grid=4)
    assert set(metrics) == {"acc_iou_0.5", "iou", "pointing", "target_mass"}
    assert all(len(values) == 2 for values in metrics.values())
    assert all(torch.isfinite(values.float()).all() for values in metrics.values())

    full_standard = GroundingDecoder(768, 768, with_ffn=True)
    full_attention = GroundingDecoder(768, 768, depth=8)
    parameter_gap = abs(
        sum(p.numel() for p in full_standard.parameters())
        - sum(p.numel() for p in full_attention.parameters())
    )
    assert parameter_gap / sum(p.numel() for p in full_standard.parameters()) < 0.003
    paired_a4 = build_decoder("A4", 48, 24, 2)
    paired_s4 = build_decoder("S4", 48, 24, 2)
    paired_shared = paired_a4.state_dict()
    for name, value in paired_s4.state_dict().items():
        if name in paired_shared and paired_shared[name].shape == value.shape:
            assert torch.equal(paired_shared[name], value), name

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "image.jpg").touch()
        (root / "instances.json").write_text(json.dumps({
            "images": [{"id": 7, "file_name": "image.jpg", "width": 100, "height": 80}],
            "annotations": [{"id": 9, "image_id": 7, "bbox": [20, 10, 40, 40]}],
        }))
        with (root / "refs.p").open("wb") as file:
            pickle.dump([{
                "ann_id": 9, "image_id": 7, "split": "train",
                "sentences": [{"sent_id": 11, "sent": "the red cup"}],
            }], file)
        count, digest, clamped = prepare(
            "refcocog", "train", root / "instances.json", root / "refs.p",
            root, root / "train.jsonl",
        )
        record = json.loads((root / "train.jsonl").read_text())
        assert count == 1 and len(digest) == 64 and not clamped
        assert record["id"] == "refcocog:11" and record["box_xyxy"] == [20.0, 10.0, 60.0, 50.0]
        assert record["stratum"] == "direct" and record["tags"] == ["attribute"]
    print("ok: taxonomy, data normalization, decoder, masks, gradients, geometry, and parameter match")


if __name__ == "__main__":
    check()
