"""Write compact CSV and Markdown views of efficiency measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    payload = json.loads(args.input.read_text()); rows = []
    for variant, row in payload["variants"].items():
        rows.append({"variant": variant, "trainable_parameters": row["trainable_parameters"], "analytical_macs_per_example": row["analytical_macs_per_example"], "analytical_flops_per_example": row["analytical_flops_per_example"], "decoder_latency_ms": 1000*row["decoder_only"]["mean_seconds"], "decoder_latency_std_ms": 1000*row["decoder_only"]["std_seconds"], "decoder_examples_per_second": row["decoder_only"]["examples_per_second"], "decoder_peak_memory_mb": row["decoder_only_peak_allocated_bytes"]/2**20, "full_pipeline_latency_ms": 1000*row["full_pipeline"]["mean_seconds"], "full_pipeline_latency_std_ms": 1000*row["full_pipeline"]["std_seconds"], "full_pipeline_examples_per_second": row["full_pipeline"]["examples_per_second"], "full_pipeline_peak_memory_mb": row["full_pipeline_peak_allocated_bytes"]/2**20, "parameters_vs_s4_percent": row["relative_to_s4"]["parameters_percent"], "macs_vs_s4_percent": row["relative_to_s4"]["macs_percent"], "decoder_latency_vs_s4_percent": row["relative_to_s4"]["decoder_latency_percent"], "full_pipeline_latency_vs_s4_percent": row["relative_to_s4"]["full_pipeline_latency_percent"]})
    args.output.parent.mkdir(parents=True, exist_ok=True); csv_path = args.output.with_suffix(".csv"); md_path = args.output.with_suffix(".md")
    with csv_path.open("w", newline="") as handle: csv.DictWriter(handle, fieldnames=list(rows[0])).writeheader(); csv.DictWriter(handle, fieldnames=list(rows[0])).writerows(rows)
    lines = ["# Decoder efficiency", "", "Cached-feature timings isolate the trainable decoder; full-pipeline timings include image/text preprocessing and the frozen SigLIP2 backbone. CUDA synchronization, warmups, and repeated measurements are recorded in `measurements.json`.", "", "| Variant | Params | MACs/example | Decoder ms | Decoder ex/s | Full pipeline ms | Full ex/s | Decoder peak MB | Full peak MB | Decoder latency vs S4 | Full latency vs S4 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows: lines.append(f"| {row['variant']} | {row['trainable_parameters']:,} | {row['analytical_macs_per_example']:,} | {row['decoder_latency_ms']:.2f} ± {row['decoder_latency_std_ms']:.2f} | {row['decoder_examples_per_second']:.2f} | {row['full_pipeline_latency_ms']:.2f} ± {row['full_pipeline_latency_std_ms']:.2f} | {row['full_pipeline_examples_per_second']:.2f} | {row['decoder_peak_memory_mb']:.1f} | {row['full_pipeline_peak_memory_mb']:.1f} | {row['decoder_latency_vs_s4_percent']:+.1f}% | {row['full_pipeline_latency_vs_s4_percent']:+.1f}% |")
    md_path.write_text("\n".join(lines)); print(md_path)


if __name__ == "__main__": main()
