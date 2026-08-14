"""Measure decoder-only and raw-image inference cost on one fixed batch."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from run import Records, analytical_macs, build_decoder, encode, load_backbone


VARIANTS = ("A4", "S4", "A8")


def timed(function, repeats: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter(); function(); torch.cuda.synchronize(); values.append(time.perf_counter() - started)
    return values


def stats(values: list[float], batch_size: int) -> dict:
    return {"mean_seconds": statistics.mean(values), "std_seconds": statistics.stdev(values) if len(values) > 1 else 0.0, "median_seconds": statistics.median(values), "p95_seconds": sorted(values)[int(0.95 * (len(values) - 1))], "examples_per_second": batch_size / statistics.mean(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("efficiency benchmark requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = Records(args.data, limit=args.batch_size)
    backbone, processor = load_backbone(checkpoint["backbone"], device)
    records = list(dataset.items)
    with torch.no_grad():
        image, text, mask, sizes, _ = encode(backbone, processor, records, device)
    image_width, text_width = image.shape[-1], text.shape[-1]
    text_length = text.shape[1]
    raw = {}
    # Decoder-only runs reuse the exact frozen features from the fixed batch.
    for variant in VARIANTS:
        decoder = build_decoder(variant, image_width, text_width, checkpoint["seed"]).to(device).eval()
        params = sum(parameter.numel() for parameter in decoder.parameters())
        macs = analytical_macs(variant, image_width, text_width, text_length)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        def decoder_forward():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                decoder(image, text, mask)
        values = timed(decoder_forward, args.repeats, args.warmups)
        raw[variant] = {"trainable_parameters": params, "analytical_macs_per_example": macs, "analytical_flops_per_example": 2 * macs, "decoder_only": stats(values, args.batch_size), "decoder_only_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        del decoder
    # Full-pipeline timings include PIL/processor work plus frozen image/text encoders.
    full = {}
    for variant in VARIANTS:
        decoder = build_decoder(variant, image_width, text_width, checkpoint["seed"]).to(device).eval()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        def forward():
            with torch.no_grad():
                image_now, text_now, mask_now, _, _ = encode(backbone, processor, records, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    decoder(image_now, text_now, mask_now)
        values = timed(forward, max(8, args.repeats // 2), max(3, args.warmups // 2))
        full[variant] = {"full_pipeline": stats(values, args.batch_size), "full_pipeline_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        del decoder
    del backbone; torch.cuda.empty_cache()
    output = {"protocol": {"backbone": checkpoint["backbone"], "checkpoint": str(args.checkpoint), "batch_size": args.batch_size, "repeats": args.repeats, "warmups": args.warmups, "cuda_synchronize": True, "fixed_manifest_prefix": str(args.data)}, "variants": {variant: raw[variant] | full[variant] for variant in VARIANTS}}
    s4 = output["variants"]["S4"]
    for variant in VARIANTS:
        row = output["variants"][variant]
        row["relative_to_s4"] = {"parameters_percent": 100 * (row["trainable_parameters"] / s4["trainable_parameters"] - 1), "macs_percent": 100 * (row["analytical_macs_per_example"] / s4["analytical_macs_per_example"] - 1), "decoder_latency_percent": 100 * (row["decoder_only"]["mean_seconds"] / s4["decoder_only"]["mean_seconds"] - 1), "full_pipeline_latency_percent": 100 * (row["full_pipeline"]["mean_seconds"] / s4["full_pipeline"]["mean_seconds"] - 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2) + "\n"); print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
