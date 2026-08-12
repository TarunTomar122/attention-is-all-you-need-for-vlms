"""Train or evaluate one frozen-backbone grounding decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from study import GroundingDecoder, box_target, copy_shared_parameters, heatmap_box, per_example_metrics


MODELS = {
    "siglip2": (
        "google/siglip2-base-patch16-384",
        "f775b65a79762255128c981547af89addcfe0f88",
    ),
    "clip": (
        "openai/clip-vit-large-patch14-336",
        "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    ),
}
VARIANTS = {
    "D0": (0, False), "A1": (1, False), "A2": (2, False),
    "A4": (4, False), "S1": (1, True), "S2": (2, True),
    "S4": (4, True), "A8": (8, False),
}


class Records(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not self.items:
            raise ValueError(f"empty manifest: {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_decoder(variant: str, image_width: int, text_width: int, seed: int) -> GroundingDecoder:
    depth, with_ffn = VARIANTS[variant]
    torch.manual_seed(seed)
    reference_depth = 4 if variant == "A8" else depth
    reference = GroundingDecoder(image_width, text_width, depth=reference_depth)
    if not with_ffn and variant != "A8":
        return reference
    model = GroundingDecoder(image_width, text_width, depth=depth, with_ffn=with_ffn)
    copy_shared_parameters(reference, model)
    return model


def load_backbone(key: str, device: torch.device):
    from transformers import AutoModel, AutoProcessor

    model_id, revision = MODELS[key]
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    backbone.requires_grad_(False).eval().to(device)
    return backbone, processor


def encode(backbone, processor, records: list[dict], device: torch.device) -> tuple[torch.Tensor, ...]:
    size = int(backbone.config.vision_config.image_size)
    resample = processor.image_processor.resample
    images = []
    for record in records:
        with Image.open(record["image"]) as image:
            images.append(image.convert("RGB").resize((size, size), resample=resample))
    image_kwargs = {"do_resize": False, "return_tensors": "pt"}
    if hasattr(processor.image_processor, "do_center_crop"):
        image_kwargs["do_center_crop"] = False
    image_inputs = processor.image_processor(images, **image_kwargs)
    max_length = int(backbone.config.text_config.max_position_embeddings)
    text_inputs = processor.tokenizer(
        [record["expression"] for record in records], padding="max_length",
        truncation=True, max_length=max_length, return_tensors="pt",
    )
    pixel_values = image_inputs["pixel_values"].to(
        device=device, dtype=next(backbone.parameters()).dtype,
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device).bool()
    with torch.no_grad():
        vision = backbone.vision_model(pixel_values=pixel_values).last_hidden_state
        text = backbone.text_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
    if vision.shape[1] == 577:  # CLIP class token
        vision = vision[:, 1:]
    if vision.shape[1] != 576:
        raise ValueError(f"expected 576 image patches, received {vision.shape[1]}")
    sizes = torch.tensor(
        [[record["width"], record["height"]] for record in records],
        dtype=torch.float32, device=device,
    )
    boxes = torch.tensor(
        [record["box_xyxy"] for record in records], dtype=torch.float32, device=device,
    )
    return vision, text, attention_mask, sizes, boxes


def batches(dataset: Records, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        collate_fn=list, num_workers=0, drop_last=shuffle,
    )


@torch.no_grad()
def validation_loss(backbone, processor, decoder, loader, device) -> float:
    decoder.eval()
    total = 0.0
    count = 0
    for records in loader:
        image, text, mask, sizes, boxes = encode(backbone, processor, records, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = decoder(image, text, mask).float()
            loss = -(box_target(boxes, sizes) * probabilities.clamp_min(1e-8).log()).sum(1)
        total += float(loss.sum())
        count += len(records)
    decoder.train()
    return total / count


def save_checkpoint(path: Path, decoder: GroundingDecoder, metadata: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"decoder": decoder.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA GPU")
    if min(args.batch_size, args.accumulate, args.steps, args.warmup_steps, args.eval_every) <= 0:
        raise ValueError("batch, accumulation, step, warm-up, and evaluation values must be positive")
    if args.warmup_steps >= args.steps:
        raise ValueError("warm-up must be shorter than the run")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, processor = load_backbone(args.backbone, device)
    decoder = build_decoder(
        args.variant, backbone.config.vision_config.hidden_size,
        backbone.config.text_config.hidden_size, args.seed,
    ).to(device)
    train_data, val_data = Records(args.train), Records(args.val)
    train_loader = batches(train_data, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = batches(val_data, args.batch_size, shuffle=False, seed=args.seed)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=0.01,
        betas=(0.9, 0.999), eps=1e-8,
    )

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    config = {name: value for name, value in vars(args).items() if name != "action"}
    metadata = {
        "variant": args.variant,
        "backbone": args.backbone,
        "model_id": MODELS[args.backbone][0],
        "model_revision": MODELS[args.backbone][1],
        "seed": args.seed,
        "image_width": backbone.config.vision_config.hidden_size,
        "text_width": backbone.config.text_config.hidden_size,
        "train_manifest_sha256": manifest_hash(args.train),
        "val_manifest_sha256": manifest_hash(args.val),
        "trainable_parameters": sum(p.numel() for p in decoder.parameters()),
        "frozen_parameters": sum(p.numel() for p in backbone.parameters()),
        "config": config,
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True,
        ).strip(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    optimizer.zero_grad(set_to_none=True)
    update = 0
    micro_step = 0
    best_loss = math.inf
    while update < args.steps:
        for records in train_loader:
            image, text, mask, sizes, boxes = encode(backbone, processor, records, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probabilities = decoder(image, text, mask).float()
                loss = -(box_target(boxes, sizes) * probabilities.clamp_min(1e-8).log()).sum(1).mean()
                loss = loss / args.accumulate
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss before update {update + 1}")
            loss.backward()
            micro_step += 1
            if micro_step % args.accumulate:
                continue
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update += 1
            if update % args.eval_every == 0 or update == args.steps:
                val_loss = validation_loss(backbone, processor, decoder, val_loader, device)
                print(json.dumps({"update": update, "val_loss": val_loss}))
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(args.output / "best.pt", decoder, metadata | {"update": update, "val_loss": val_loss})
            if update >= args.steps:
                break


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone, processor = load_backbone(checkpoint["backbone"], device)
    decoder = build_decoder(
        checkpoint["variant"], checkpoint["image_width"], checkpoint["text_width"], checkpoint["seed"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    dataset = Records(args.data)
    heatmaps, predicted_boxes, target_boxes, image_sizes = [], [], [], []
    ids, image_ids, strata, tags, compositional, token_counts = [], [], [], [], [], []
    with torch.no_grad():
        for records in batches(dataset, args.batch_size, shuffle=False, seed=checkpoint["seed"]):
            image, text, mask, sizes, boxes = encode(backbone, processor, records, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probabilities = decoder(image, text, mask).float()
            heatmaps.append(probabilities.cpu())
            predicted_boxes.append(heatmap_box(probabilities, sizes, args.mass).cpu())
            target_boxes.append(boxes.cpu())
            image_sizes.append(sizes.cpu())
            ids.extend(record["id"] for record in records)
            image_ids.extend(record["image_id"] for record in records)
            strata.extend(record["stratum"] for record in records)
            tags.extend(record["tags"] for record in records)
            compositional.extend(record["compositional"] for record in records)
            token_counts.extend(record["token_count"] for record in records)
    heatmaps_tensor = torch.cat(heatmaps).float()
    predicted_tensor = torch.cat(predicted_boxes)
    target_tensor = torch.cat(target_boxes)
    sizes_tensor = torch.cat(image_sizes)
    metrics = per_example_metrics(
        heatmaps_tensor, predicted_tensor, target_tensor, sizes_tensor,
    )
    result = {
        "ids": ids, "image_ids": image_ids, "strata": strata, "tags": tags,
        "compositional": compositional, "token_counts": token_counts,
        "heatmaps": heatmaps_tensor, "predicted_boxes": predicted_tensor,
        "target_boxes": target_tensor, "image_sizes": sizes_tensor,
        "metrics": metrics, "mass": args.mass,
        "manifest_sha256": manifest_hash(args.data),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({name: float(values.float().mean()) for name, values in metrics.items()}))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--train", required=True, type=Path)
    training.add_argument("--val", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--backbone", choices=MODELS, default="siglip2")
    training.add_argument("--variant", choices=VARIANTS, required=True)
    training.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    training.add_argument("--learning-rate", type=float, required=True)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--accumulate", type=int, default=2)
    training.add_argument("--steps", type=int, default=5000)
    training.add_argument("--warmup-steps", type=int, default=250)
    training.add_argument("--eval-every", type=int, default=500)
    training.set_defaults(action=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--data", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--mass", type=float, required=True)
    evaluation.set_defaults(action=evaluate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.action(arguments)
