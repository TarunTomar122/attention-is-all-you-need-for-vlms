"""Minimal FFN-free visual-grounding study primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn


WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
LEXICONS = {
    "attribute": {
        "beige", "black", "blue", "brown", "colorful", "dark", "gold",
        "gray", "green", "grey", "light", "orange", "pink", "purple",
        "red", "silver", "striped", "white", "wooden", "yellow",
    },
    "absolute": {
        "background", "bottom", "bottommost", "center", "central",
        "foreground", "leftmost", "middle", "rear", "rightmost", "top",
        "topmost",
    },
    "comparison": {
        "biggest", "closest", "farthest", "fewest", "furthest", "highest",
        "largest", "least", "longest", "lowest", "most", "nearest", "same",
        "shortest", "smaller", "smallest", "tallest",
    },
    "ordinal": {
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth",
    },
    "cardinality": {
        "both", "couple", "double", "eight", "five", "four", "nine", "pair",
        "seven", "six", "three", "triple", "two",
    },
    "negation": {
        "aren't", "cannot", "can't", "doesn't", "don't", "except", "isn't",
        "neither", "no", "nor", "not", "nothing", "without",
    },
}
RELATIONS = (
    "attached to", "behind", "below", "beside", "between", "carrying",
    "close to", "covered by", "crossing", "facing", "far from", "holding",
    "in front of", "inside", "left of", "looking at", "near", "next to",
    "on top of", "outside", "owned by", "ridden by", "riding", "right of",
    "under", "underneath", "wearing", "with", "worn by",
)
ABSOLUTE_PHRASES = (
    "at the left", "at the right", "in the left", "in the right",
    "on the left", "on the right",
)


@dataclass(frozen=True)
class Classification:
    stratum: str
    tags: tuple[str, ...]
    relation_count: int
    token_count: int
    compositional: bool


def classify_expression(expression: str) -> Classification:
    """Assign observable lexical tags without inspecting images or predictions."""
    normalized = " ".join(WORDS.findall(expression.lower()))
    tokens = normalized.split()
    token_set = set(tokens)
    tags = {name for name, words in LEXICONS.items() if token_set & words}
    if any(token.isdigit() for token in tokens):
        tags.add("cardinality")
    if any(phrase in normalized for phrase in ABSOLUTE_PHRASES):
        tags.add("absolute")
    relation_count = sum(normalized.count(phrase) for phrase in RELATIONS)
    if relation_count:
        tags.add("relation")

    if tags & {"comparison", "ordinal", "cardinality", "negation"}:
        stratum = "logical"
    elif "relation" in tags:
        stratum = "relational"
    elif "absolute" in tags:
        stratum = "absolute"
    else:
        stratum = "direct"

    structural_tags = tags - {"attribute"}
    return Classification(
        stratum=stratum,
        tags=tuple(sorted(tags)),
        relation_count=relation_count,
        token_count=len(tokens),
        compositional=relation_count >= 2 or len(structural_tags) >= 2,
    )


def _rms_norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


class CrossAttention(nn.Module):
    """Cross-attention with non-affine per-head Q/K RMS normalization."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(
        self, query: Tensor, context: Tensor, valid_mask: Tensor | None = None,
        *, return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch, query_length, width = query.shape
        context_length = context.shape[1]
        shape_q = (batch, query_length, self.heads, self.head_width)
        shape_kv = (batch, context_length, self.heads, self.head_width)
        q = _rms_norm(self.q(query).view(shape_q)).transpose(1, 2)
        k = _rms_norm(self.k(context).view(shape_kv)).transpose(1, 2)
        v = self.v(context).view(shape_kv).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / self.head_width**0.5
        if valid_mask is not None:
            if valid_mask.shape != (batch, context_length) or not valid_mask.any(1).all():
                raise ValueError("valid_mask must retain at least one context token per row")
            logits = logits.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
        weights = logits.softmax(dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, query_length, width)
        output = self.out(attended)
        return (output, weights) if return_weights else output


class FFN(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class GroundingBlock(nn.Module):
    def __init__(self, width: int, heads: int, with_ffn: bool) -> None:
        super().__init__()
        self.text_query_norm = nn.LayerNorm(width)
        self.text_context_norm = nn.LayerNorm(width)
        self.text_attention = CrossAttention(width, heads)
        self.image_query_norm = nn.LayerNorm(width)
        self.image_context_norm = nn.LayerNorm(width)
        self.image_attention = CrossAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width) if with_ffn else None
        self.ffn = FFN(width) if with_ffn else None

    def forward(self, query: Tensor, text: Tensor, image: Tensor, text_mask: Tensor) -> Tensor:
        query = query + self.text_attention(
            self.text_query_norm(query), self.text_context_norm(text), text_mask,
        )
        query = query + self.image_attention(
            self.image_query_norm(query), self.image_context_norm(image),
        )
        if self.ffn is not None and self.ffn_norm is not None:
            query = query + self.ffn(self.ffn_norm(query))
        return query


class AttentionReadout(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)

    def forward(self, query: Tensor, image: Tensor) -> Tensor:
        batch, patches, _ = image.shape
        q = _rms_norm(self.q(query).view(batch, 1, self.heads, self.head_width))
        k = _rms_norm(self.k(image).view(batch, patches, self.heads, self.head_width))
        logits = torch.einsum("bqhd,bphd->bhqp", q, k) / self.head_width**0.5
        return logits.mean(dim=1).squeeze(1).softmax(dim=-1)


class GroundingDecoder(nn.Module):
    """One-query grounding decoder over already-frozen image/text tokens."""

    def __init__(
        self, image_width: int, text_width: int, *, width: int = 256,
        heads: int = 8, depth: int = 4, with_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.image_projection = nn.Linear(image_width, width)
        self.text_projection = nn.Linear(text_width, width)
        self.query = nn.Parameter(torch.empty(1, 1, width)) if depth else None
        if self.query is not None:
            nn.init.normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList(
            GroundingBlock(width, heads, with_ffn) for _ in range(depth)
        )
        self.readout = AttentionReadout(width, heads)

    def forward(self, image_tokens: Tensor, text_tokens: Tensor, text_mask: Tensor) -> Tensor:
        if (
            text_mask.dtype is not torch.bool
            or text_mask.shape != text_tokens.shape[:2]
            or not text_mask.any(1).all()
        ):
            raise ValueError("text_mask must mark at least one valid token per row")
        image = self.image_projection(image_tokens)
        text = self.text_projection(text_tokens)
        if self.query is None:
            query = (text * text_mask.unsqueeze(-1)).sum(1, keepdim=True)
            query = query / text_mask.sum(1, keepdim=True).unsqueeze(-1)
        else:
            query = self.query.expand(image.shape[0], -1, -1)
            for block in self.blocks:
                query = block(query, text, image, text_mask)
        return self.readout(query, image)


def copy_shared_parameters(source: nn.Module, target: nn.Module) -> None:
    """Copy every name-and-shape-compatible tensor for paired initialization."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone()
    target.load_state_dict(target_state)


def _box_overlap(boxes_xyxy: Tensor, image_sizes_wh: Tensor, grid: int) -> tuple[Tensor, Tensor]:
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [batch, 4]")
    if image_sizes_wh.shape != (boxes_xyxy.shape[0], 2):
        raise ValueError("image_sizes_wh must have shape [batch, 2]")
    in_bounds = (
        (boxes_xyxy[:, :2] >= 0).all()
        and (boxes_xyxy[:, 2:] <= image_sizes_wh).all()
    )
    if not (
        (boxes_xyxy[:, 2:] > boxes_xyxy[:, :2]).all()
        and (image_sizes_wh > 0).all()
        and in_bounds
    ):
        raise ValueError("boxes must have positive area within positive image bounds")

    steps = torch.arange(grid, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    width, height = image_sizes_wh[:, 0], image_sizes_wh[:, 1]
    x0 = steps[None, :] * width[:, None] / grid
    x1 = (steps[None, :] + 1) * width[:, None] / grid
    y0 = steps[None, :] * height[:, None] / grid
    y1 = (steps[None, :] + 1) * height[:, None] / grid
    overlap_x = (
        torch.minimum(x1, boxes_xyxy[:, 2, None])
        - torch.maximum(x0, boxes_xyxy[:, 0, None])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(y1, boxes_xyxy[:, 3, None])
        - torch.maximum(y0, boxes_xyxy[:, 1, None])
    ).clamp_min(0)
    area = overlap_y[:, :, None] * overlap_x[:, None, :]
    patch_area = (width * height / grid**2)[:, None, None]
    return area, patch_area


def box_target(boxes_xyxy: Tensor, image_sizes_wh: Tensor, grid: int = 24) -> Tensor:
    """Convert original-image boxes to area-weighted grid distributions."""
    area, _ = _box_overlap(boxes_xyxy, image_sizes_wh, grid)
    return (area / area.sum((1, 2), keepdim=True)).flatten(1)


def _shortest_interval(marginal: Tensor, mass: float) -> tuple[int, int]:
    # ponytail: O(grid²) is simpler and grid=24; optimize only for larger grids.
    best: tuple[int, float, int, int] | None = None
    for start in range(len(marginal)):
        total = 0.0
        for end in range(start, len(marginal)):
            total += float(marginal[end])
            if total >= mass:
                candidate = (end - start + 1, -total, start, end + 1)
                best = candidate if best is None or candidate < best else best
                break
    if best is None:
        raise ValueError("heatmap does not contain requested finite mass")
    return best[2], best[3]


def heatmap_box(probabilities: Tensor, image_sizes_wh: Tensor, mass: float, grid: int = 24) -> Tensor:
    """Convert grid distributions to original-image boxes deterministically."""
    valid_distribution = (
        probabilities.ndim == 2
        and probabilities.shape[1] == grid * grid
        and torch.isfinite(probabilities).all()
        and (probabilities >= 0).all()
        and torch.allclose(
            probabilities.sum(1), torch.ones(len(probabilities), device=probabilities.device),
            atol=1e-4,
        )
    )
    if not valid_distribution or not 0 < mass <= 1:
        raise ValueError("invalid heatmap shape or mass")
    maps = probabilities.detach().float().cpu().reshape(-1, grid, grid)
    boxes = []
    for heatmap, (width, height) in zip(maps, image_sizes_wh.detach().cpu()):
        left, right = _shortest_interval(heatmap.sum(0), mass)
        top, bottom = _shortest_interval(heatmap.sum(1), mass)
        boxes.append((
            left * float(width) / grid, top * float(height) / grid,
            right * float(width) / grid, bottom * float(height) / grid,
        ))
    return torch.tensor(boxes, dtype=probabilities.dtype, device=probabilities.device)


def per_example_metrics(
    probabilities: Tensor, predicted_boxes: Tensor, target_boxes: Tensor,
    image_sizes_wh: Tensor, grid: int = 24,
) -> dict[str, Tensor]:
    """Return raw per-example metrics used by the preregistered analysis."""
    intersection_left_top = torch.maximum(predicted_boxes[:, :2], target_boxes[:, :2])
    intersection_right_bottom = torch.minimum(predicted_boxes[:, 2:], target_boxes[:, 2:])
    intersection = (intersection_right_bottom - intersection_left_top).clamp_min(0).prod(1)
    predicted_area = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(1)
    target_area = (target_boxes[:, 2:] - target_boxes[:, :2]).prod(1)
    iou = intersection / (predicted_area + target_area - intersection).clamp_min(1e-8)

    peak = probabilities.argmax(1)
    peak_x = (peak % grid + 0.5) * image_sizes_wh[:, 0] / grid
    peak_y = (peak // grid + 0.5) * image_sizes_wh[:, 1] / grid
    pointing = (
        (peak_x >= target_boxes[:, 0]) & (peak_x <= target_boxes[:, 2])
        & (peak_y >= target_boxes[:, 1]) & (peak_y <= target_boxes[:, 3])
    )
    overlap, patch_area = _box_overlap(target_boxes, image_sizes_wh, grid)
    target_mass = (probabilities.view(-1, grid, grid) * overlap / patch_area).sum((1, 2))
    return {
        "acc_iou_0.5": iou >= 0.5,
        "iou": iou,
        "pointing": pointing,
        "target_mass": target_mass,
    }
