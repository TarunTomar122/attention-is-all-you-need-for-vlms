#!/usr/bin/env python3
"""Render the frozen evidence into a reviewable PDF without a GPU or LaTeX."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "main.pdf"
rl_config.invariant = True
INK = colors.HexColor("#17211b")
BLUE = colors.HexColor("#2563eb")
RED = colors.HexColor("#dc2626")
GREEN = colors.HexColor("#15803d")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#dbe3ec")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=INK, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", parent=base["BodyText"], fontName="Helvetica", fontSize=10.5, leading=14, textColor=MUTED, alignment=TA_CENTER, spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=INK, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.7, leading=10, textColor=MUTED, spaceAfter=4),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_LEFT, spaceAfter=8),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.2, leading=14.2, textColor=INK, spaceAfter=0),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName="Helvetica", fontSize=7.25, leading=9, textColor=INK),
        "tablehead": ParagraphStyle("tablehead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.25, leading=9, textColor=colors.white),
    }


def figure(path: Path, width: float) -> Image:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE); canvas.line(doc.leftMargin, 0.42 * inch, letter[0] - doc.rightMargin, 0.42 * inch)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.27 * inch, "Do Visual Grounding Decoders Need Feed-Forward Networks?")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.27 * inch, f"{doc.page}")
    canvas.restoreState()


def make_table(rows: list[list[str]], widths: list[float], s: dict[str, ParagraphStyle], header: bool = True) -> Table:
    body = [[Paragraph(cell, s["tablehead"] if header and index == 0 else s["table"]) for cell in row] for index, row in enumerate(rows)]
    table = Table(body, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), INK))
    table.setStyle(TableStyle(commands))
    return table


def main() -> None:
    s = styles()
    data = json.loads((PAPER / "data/paper-data.json").read_text())
    fine = data["results"]["finecops"]
    adv = data["results"]["refadv"]
    with (PAPER / "tables/generated-main-results.csv").open(newline="") as handle:
        main_rows = list(csv.reader(handle))
    with (ROOT / "docs/results/finecops/finecops_slices.csv").open(newline="") as handle:
        slices = list(csv.DictReader(handle))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=.63 * inch, rightMargin=.63 * inch, topMargin=.58 * inch, bottomMargin=.62 * inch, title=data["title"], author="Tarun Tomar")
    story = []
    story += [Paragraph(data["title"], s["title"]), Paragraph("Tarun Tomar | University of Edinburgh<br/>A controlled study over frozen vision-language features", s["subtitle"])]
    story += [Paragraph("Abstract", s["h1"])]
    abstract = "Do feed-forward networks (FFNs) in visual grounding decoders add essential computation once a pretrained vision-language model has already encoded image and language context? We compare a four-block attention-only decoder (A4), a matched four-block attention-plus-FFN decoder (S4), and an eight-block attention-only parameter control (A8) over frozen VLM features. A4 matches or slightly exceeds S4 on RefCOCOg and Ref-Adv-s. FineCops-Ref reveals a small A4 deficit (-0.52 pp IoU@0.5, 95% CI [-0.95, -0.12]), but A8 recovers it (+0.26 pp versus S4). Official FineCops levels do not show a monotonic increase in the gap. A4 reduces trainable decoder parameters by 44.4% and cached-decoder latency by 10.1%, while the frozen VLM leaves end-to-end latency essentially unchanged. The claim is intentionally narrow: FFNs are often dispensable in this small grounding decoder, not in complete VLMs generally."
    story += [Paragraph(abstract, s["body"]), Spacer(1, 5)]
    callout = Table([[Paragraph("Main result: a small fixed-depth FineCops gap is recovered by reallocating the FFN parameter budget into attention depth. There is no supported monotonic hard-example boundary.", s["callout"])]], colWidths=[7.2 * inch])
    callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")), ("BOX", (0, 0), (-1, -1), 0.8, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [callout, Spacer(1, 13), Paragraph("1. Intervention", s["h1"])]
    story += [Paragraph("The primary frozen SigLIP2 encoder produces 576 image tokens in a 24x24 raster grid and contextual text states. Shared linear maps project both streams to width 256. One learned query alternates text and image cross-attention. S4 adds a pre-normalized GELU FFN after each pair; A4 omits it. A8 doubles attention-only depth to approximately match S4's trainable parameter count. The shared patch-attention readout, supervision, optimizer, data order, and heatmap-to-box conversion are held fixed.", s["body"])]
    story += [figure(PAPER / "figures/generated-method-overview.png", 7.15 * inch), Paragraph("Figure 1. The frozen VLM stays unchanged. The only causal intervention is the trainable decoder's FFN residual; A8 tests capacity reallocation rather than a compute-matched alternative.", s["caption"])]
    story += [PageBreak(), Paragraph("2. Evaluation contract", s["h1"])]
    story += [Paragraph("The heatmap-to-box mass tau=0.8 was selected once on RefCOCOg validation and frozen before every held-out or out-of-distribution evaluation. Three-seed comparisons average paired seeds per example and use 10,000 image-clustered bootstrap replicates with locked seed 20260812. FineCops uses only official level and tuple-type labels; Ref-Adv-s uses native released metadata only.", s["body"])]
    story += [Paragraph("3. Evidence", s["h1"])]
    story += [make_table(main_rows, [1.62 * inch, 1.05 * inch, 1.32 * inch, 3.2 * inch], s), Spacer(1, 10)]
    story += [Paragraph("RefCOCOg and modality controls", s["h2"]), Paragraph("On RefCOCOg direct expressions, A4-S4 is +0.26 pp (90% CI [-0.33, +0.84]). The direct-retention gate passes, but the predeclared direct-minus-logical interaction is not confirmed. Image and text shuffles substantially lower both decoders, supporting genuine dependence on both modalities rather than a one-modality shortcut.", s["body"])]
    story += [Paragraph("Adversarial grounding", s["h2"]), Paragraph(f"On Ref-Adv-s, A4-S4 is {adv['a4_minus_s4_pp']:+.2f} pp (95% CI [{adv['ci95_pp'][0]:+.2f}, {adv['ci95_pp'][1]:+.2f}]). Negation, expression-length, and distractor-count slices do not show a monotonic A4 loss. This falsifies the expected broad collapse without proving equality in every small slice.", s["body"])]
    story += [figure(ROOT / "docs/results/refadv/refadv_delta_by_difficulty.png", 6.65 * inch), Paragraph("Figure 2. Ref-Adv-s paired A4-S4 deltas by preregistered/native difficulty metadata. There is no monotonic loss with expression length or distractor count.", s["caption"])]
    story += [PageBreak(), Paragraph("4. Controlled compositional grounding", s["h1"])]
    story += [Paragraph(f"FineCops-Ref is the controlled test that reveals the only clear fixed-depth deficit: A4-S4 is {fine['a4_minus_s4_pp']:+.2f} pp (95% CI [{fine['ci95_pp'][0]:+.2f}, {fine['ci95_pp'][1]:+.2f}]). A8-S4 is {fine['a8_minus_s4_pp']:+.2f} pp. The official level-3 estimate is noisy and slightly favors A4, which means the study does not support a monotonic harder-means-more-FFN-needed interpretation.", s["body"])]
    story += [figure(PAPER / "figures/generated-finecops-difficulty.png", 7.15 * inch), Paragraph("Figure 3. FineCops official difficulty levels. A4 has a small overall deficit, while A8 recovers above S4 overall; level 3 is too imprecise to establish a monotonic boundary.", s["caption"])]
    level_rows = [["Level", "N", "A4", "S4", "A8", "A4-S4 95% CI"]]
    for row in slices:
        if row["slice"].startswith("level_"):
            level_rows.append([row["slice"].replace("level_", ""), row["n"], f"{100*float(row['A4']):.2f}%", f"{100*float(row['S4']):.2f}%", f"{100*float(row['A8']):.2f}%", f"[{100*float(row['a4_s4_ci95_low']):+.2f}, {100*float(row['a4_s4_ci95_high']):+.2f}]"])
    story += [make_table(level_rows, [.55 * inch, .55 * inch, .75 * inch, .75 * inch, .75 * inch, 1.55 * inch], s), Spacer(1, 10)]
    story += [Paragraph("Qualitative-panel provenance", s["h2"]), Paragraph("The versioned repository contains aggregate/per-example metrics but not the raw target and predicted box coordinates or licensed image files needed to make honest qualitative panels. This PDF therefore includes only reproducible architecture and aggregate evidence figures. Add qualitative examples only from the preserved raw prediction exports, with source attribution; do not recreate them from aggregate tables.", s["small"])]
    story += [PageBreak(), Paragraph("5. Efficiency, limits, and conclusion", s["h1"])]
    story += [figure(PAPER / "figures/generated-efficiency-summary.png", 6.65 * inch), Paragraph("Figure 4. A4 is substantially smaller and lower-latency as a cached-feature decoder. A8 is parameter-matched to S4, not compute-matched.", s["caption"])]
    story += [Paragraph("Efficiency", s["h2"]), Paragraph("A4 has 2.64M trainable decoder parameters compared with S4's 4.74M. Cached decoder latency is 6.71 ms versus 7.46 ms. Full-pipeline latency is 209.35 ms versus 210.20 ms: the frozen VLM dominates, so efficiency is supporting evidence rather than the headline.", s["body"])]
    story += [Paragraph("Limits", s["h2"]), Paragraph("This is an FFN-free trainable grounding decoder over frozen VLM features, not an attention-only VLM. The study does not test fine-tuning, pixel segmentation, multiple object queries, or non-frozen backbones. The RefCOCO seed-0 and CLIP one-seed results are descriptive. A8 recovery shows that attention capacity can be a viable replacement here; it does not prove the mechanism behind every FFN effect.", s["body"])]
    story += [Paragraph("Conclusion", s["h2"]), Paragraph("FFN deletion is surprisingly benign for this small grounding decoder on standard and adversarial referring-expression grounding. A small controlled-compositional deficit appears at fixed depth, but attention depth recovers it and no monotonic failure boundary emerges. The bounded architectural conclusion is that when a frozen VLM already provides relevant visual and linguistic context, attention-only decoder capacity can often substitute for FFN capacity for single-query grounding.", s["body"])]
    story += [Paragraph("Reproducibility", s["h2"]), Paragraph("This project releases scripts, generated figures/tables, result summaries, per-example metrics, bootstrap inputs, and decision logs. Raw licensed images and provider-local checkpoints are excluded. The full source audit is in paper/citation-audit.md.", s["body"])]
    story += [Paragraph("References", s["h2"])]
    references = [
        "Dong et al. Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks. ICLR, 2026.",
        "Ding et al. Vision-Language Transformer and Query Generation for Referring Segmentation. ICCV, 2021.",
        "Kang et al. Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding. CVPR, 2025.",
        "Liu et al. FineCops-Ref: A New Dataset and Task for Fine-Grained Compositional Referring Expression Comprehension. arXiv:2409.14750, 2024.",
        "Ndubuaku et al. A Controlled Study of Attention-Only Transformers. arXiv:2607.18363, 2026.",
        "Tschannen et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786, 2025.",
        "Wu et al. F-LMM: Grounding Frozen Large Multimodal Models. CVPR, 2025.",
        "Yu et al. Modeling Context in Referring Expressions. ECCV, 2016.",
    ]
    story += [Paragraph(f"[{index}] {reference}", s["small"]) for index, reference in enumerate(references, start=1)]
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUT)


if __name__ == "__main__":
    main()
