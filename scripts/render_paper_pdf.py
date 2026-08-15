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
    story += [Paragraph(data["title"], s["title"]), Paragraph("Tarun Tomar<br/>A controlled study over frozen vision-language features", s["subtitle"])]
    story += [Paragraph("Abstract", s["h1"])]
    abstract = "Do feed-forward networks (FFNs) in visual grounding decoders add essential computation once a pretrained vision-language model has already encoded image and language context? We compare a four-block attention-only decoder (A4), a matched four-block attention-plus-FFN decoder (S4), and an eight-block attention-only parameter control (A8) over frozen VLM features. A4 matches or slightly exceeds S4 on RefCOCOg and Ref-Adv-s. FineCops-Ref reveals a small A4 deficit (-0.52 pp IoU@0.5, 95% CI [-0.95, -0.12]), but A8 recovers it (+0.26 pp versus S4). Official FineCops levels do not show a monotonic increase in the gap. A4 reduces trainable decoder parameters by 44.4% and cached-decoder latency by 10.1%, while the frozen VLM leaves end-to-end latency essentially unchanged. The claim is intentionally narrow: FFNs are often dispensable in this small grounding decoder, not in complete VLMs generally."
    story += [Paragraph(abstract, s["body"]), Spacer(1, 5)]
    callout = Table([[Paragraph("Main result: a small fixed-depth FineCops gap is recovered by reallocating the FFN parameter budget into attention depth. There is no supported monotonic hard-example boundary.", s["callout"])]], colWidths=[7.2 * inch])
    callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")), ("BOX", (0, 0), (-1, -1), 0.8, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [callout, Spacer(1, 13), Paragraph("1. Introduction", s["h1"])]
    story += [Paragraph("Visual grounding maps an image and a referring expression to one target box. Standard Transformer grounding heads retain an attention-plus-FFN block, even when their input has already been contextualized by a large pretrained vision-language model. This creates a specific architectural question: after the frozen model supplies visual locations and language context, is token-wise FFN computation still necessary in the small trainable head?", s["body"])]
    story += [Paragraph("The question is not whether attention maps can localize objects. Existing frozen-LVLM work already shows that a few internal attention heads can be useful for grounding. It is also not whether complete VLMs can delete FFNs. This study isolates one trainable component and asks what changes when its FFN residuals are removed under a matched decoder, data, supervision, post-processing, and evaluation contract.", s["body"])]
    story += [Paragraph("We study three questions:", s["body"])]
    question_rows = [["1", "At fixed decoder depth, does A4 retain the grounding performance of S4?"], ["2", "If a gap appears, does it grow with released/official grounding difficulty metadata?"], ["3", "Can additional attention depth recover a gap after the FFN parameter budget is reallocated?"]]
    story += [make_table(question_rows, [.38 * inch, 6.82 * inch], s, header=False)]
    story += [Paragraph("The answer is mixed but coherent. A4 broadly matches S4 on RefCOCOg and Ref-Adv-s. FineCops-Ref reveals a small fixed-depth deficit, but A8 recovers it. The difficulty slices do not support the simpler story that attention-only decoding increasingly fails as references become harder.", s["body"])]

    story += [PageBreak(), Paragraph("2. Why the comparison is non-trivial", s["h1"])]
    story += [Paragraph("Attention can route a query to relevant image and text tokens. An FFN instead applies a token-wise nonlinear transformation after routing. When the relevant visual and language context already exists in frozen VLM features, it is plausible that a small decoder primarily needs retrieval and assembly. But this is not guaranteed: relations, compositional attributes, or distractor resolution might require transformations that attention-only depth cannot reproduce at the same scale.", s["body"])]
    story += [Paragraph("A global score alone cannot answer this question. A4 could match S4 because the dataset is dominated by direct references, while losing on expressions involving relations or compositional constraints. Conversely, an apparent hard-example loss could be caused by a changed data distribution, a threshold selected after test access, a position prior, or a one-modality shortcut. The study therefore uses frozen selection rules, paired seeds, modality shuffles, a released adversarial set, and a controlled compositional benchmark.", s["body"])]
    story += [Paragraph("The control logic is deliberately narrow. S4 versus A4 isolates fixed-depth FFN deletion. A8 then tests a different question: whether the trainable capacity removed with the FFNs can be reallocated into additional attention depth. A8 is approximately parameter-matched to S4, not compute-matched, so it cannot establish a universal efficiency advantage.", s["body"])]

    story += [PageBreak(), Paragraph("3. Decoder intervention", s["h1"])]
    story += [Paragraph("The primary frozen SigLIP2 encoder produces 576 image tokens in a 24x24 raster grid and contextual text states. Shared linear maps project both streams to width 256. A single learned grounding query alternates text and image cross-attention. S4 adds a pre-normalized GELU FFN after each pair; A4 omits it. A8 doubles attention-only depth to approximately match S4's trainable parameter count. The shared patch-attention readout, supervision, optimizer, data order, and heatmap-to-box conversion are held fixed.", s["body"])]
    story += [Paragraph("For a query state q, text tokens T, and image tokens I, every decoder block applies q <- q + CrossAttentionText(LayerNorm(q), LayerNorm(T)) followed by q <- q + CrossAttentionImage(LayerNorm(q), LayerNorm(I)). S4 then applies q <- q + FFN(LayerNorm(q)); A4 does not. The readout maps the final query and image tokens to one softmax-normalized patch distribution. Training minimizes cross-entropy against a normalized patch-overlap target. There is no coordinate-regression MLP, segmentation refiner, auxiliary loss, multi-query head, or backbone adaptation.", s["body"])]
    story += [figure(PAPER / "figures/generated-method-overview.png", 7.15 * inch), Paragraph("Figure 1. The frozen VLM stays unchanged. The only causal intervention is the trainable decoder's FFN residual; A8 tests capacity reallocation rather than a compute-matched alternative.", s["caption"])]

    story += [PageBreak(), Paragraph("4. Experimental protocol", s["h1"])]
    story += [Paragraph("The heatmap-to-box mass tau=0.8 was selected once on RefCOCOg validation and frozen before every held-out or out-of-distribution evaluation. Three-seed comparisons average paired seeds per example and use 10,000 image-clustered bootstrap replicates with locked seed 20260812. Image-level resampling retains all expressions for a sampled image, avoiding an independence assumption for repeated image references.", s["body"])]
    dataset_rows = [["Evaluation", "Role", "Scope"], ["RefCOCOg UMD", "Primary", "Three paired seeds, direct/logical protocol, modality shuffles."], ["RefCOCO UNC", "Replication", "Seed-0 descriptive classic-dataset batch."], ["Ref-Adv-s", "Adversarial/OOD", "1,142 prepared rows; evaluation only, native metadata slices."], ["FineCops-Ref", "Compositional", "9,605 official positive-test rows; official level and tuple fields only."]]
    story += [make_table(dataset_rows, [1.25 * inch, 1.1 * inch, 4.85 * inch], s), Spacer(1, 8)]
    story += [Paragraph("The primary outcome is IoU@0.5. Mean IoU, pointing accuracy, and target mass remain secondary. Ref-Adv-s length and distractor bins are inclusive empirical metadata quartiles determined before performance slicing. FineCops is never retuned: its official level and tuple-type annotations are reported as released. No LLM-derived semantic labels are added to either set.", s["body"])]
    story += [Paragraph("The confirmatory RefCOCOg gate requires practical direct-reference retention and a direct-minus-logical interaction. The direct retention condition passes, but the interaction interval crosses zero. The paper therefore treats the primary result as a bounded retention result, not a confirmed retrieval-versus-reasoning boundary.", s["body"])]

    story += [PageBreak(), Paragraph("5. Results across standard and adversarial grounding", s["h1"])]
    story += [make_table(main_rows, [1.62 * inch, 1.05 * inch, 1.32 * inch, 3.2 * inch], s), Spacer(1, 10)]
    story += [Paragraph("RefCOCOg and modality controls", s["h2"]), Paragraph("On RefCOCOg direct expressions, A4-S4 is +0.26 pp (90% CI [-0.33, +0.84]). The direct-retention gate passes, but the predeclared direct-minus-logical interaction is not confirmed. Correct-pair predictions exceed both image-shuffle and text-shuffle conditions for A4 and S4. Averaged over test examples and seeds, correct minus image-shuffle is +48.7 pp for A4 and +48.6 pp for S4; correct minus text-shuffle is +22.6 pp and +22.4 pp. These diagnostics rule out the simplest one-modality shortcut explanation without establishing a semantic-reasoning boundary.", s["body"])]
    story += [Paragraph("Adversarial grounding", s["h2"]), Paragraph(f"On Ref-Adv-s, A4-S4 is {adv['a4_minus_s4_pp']:+.2f} pp (95% CI [{adv['ci95_pp'][0]:+.2f}, {adv['ci95_pp'][1]:+.2f}]). Negation, expression-length, and distractor-count slices do not show a monotonic A4 loss. The longest-expression quartile is +1.79 pp and the highest-distractor quartile is 0.00 pp. This rejects an expected broad collapse on the released metadata, without proving equality in every low-count slice or every unobserved reasoning category.", s["body"])]
    story += [figure(ROOT / "docs/results/refadv/refadv_delta_by_difficulty.png", 6.65 * inch), Paragraph("Figure 2. Ref-Adv-s paired A4-S4 deltas by preregistered/native difficulty metadata. There is no monotonic loss with expression length or distractor count.", s["caption"])]

    story += [PageBreak(), Paragraph("6. Controlled compositional grounding", s["h1"])]
    story += [Paragraph(f"FineCops-Ref is the controlled test that reveals the only clear fixed-depth deficit: A4-S4 is {fine['a4_minus_s4_pp']:+.2f} pp (95% CI [{fine['ci95_pp'][0]:+.2f}, {fine['ci95_pp'][1]:+.2f}]). A8-S4 is {fine['a8_minus_s4_pp']:+.2f} pp. The fixed-depth gap is statistically compatible with a useful role for FFNs on this compositional benchmark. It is not a general necessity result, because the additional attention-depth control returns to and slightly exceeds S4 overall.", s["body"])]
    story += [figure(PAPER / "figures/generated-finecops-difficulty.png", 7.15 * inch), Paragraph("Figure 3. FineCops official difficulty levels. A4 has a small overall deficit, while A8 recovers above S4 overall; level 3 is too imprecise to establish a monotonic boundary.", s["caption"])]
    level_rows = [["Level", "N", "A4", "S4", "A8", "A4-S4 95% CI"]]
    for row in slices:
        if row["slice"].startswith("level_"):
            level_rows.append([row["slice"].replace("level_", ""), row["n"], f"{100*float(row['A4']):.2f}%", f"{100*float(row['S4']):.2f}%", f"{100*float(row['A8']):.2f}%", f"[{100*float(row['a4_s4_ci95_low']):+.2f}, {100*float(row['a4_s4_ci95_high']):+.2f}]"])
    story += [make_table(level_rows, [.55 * inch, .55 * inch, .75 * inch, .75 * inch, .75 * inch, 1.55 * inch], s), Spacer(1, 10)]
    story += [Paragraph("Qualitative-panel provenance", s["h2"]), Paragraph("The versioned repository contains aggregate/per-example metrics but not the raw target and predicted box coordinates or licensed image files needed to make honest qualitative panels. This manuscript therefore includes only reproducible architecture and aggregate evidence figures. Add qualitative examples only from the preserved raw prediction exports, with source attribution; do not recreate them from aggregate tables.", s["small"])]

    story += [PageBreak(), Paragraph("7. Controls and efficiency", s["h1"])]
    story += [Paragraph("The causal comparison is strongest when the rest of the pipeline is shared. D0, uniform, and position-prior baselines show that the trained decoders outperform non-grounded outputs. Modality shuffles show that both A4 and S4 depend on both image and text. A RefCOCO seed-0 batch checks the result on a classic dataset, while the frozen CLIP one-seed control has A4-S4 = +0.18 pp. These latter controls are descriptive rather than multi-seed confirmation evidence.", s["body"])]
    story += [figure(PAPER / "figures/generated-efficiency-summary.png", 6.65 * inch), Paragraph("Figure 4. A4 is substantially smaller and lower-latency as a cached-feature decoder. A8 is parameter-matched to S4, not compute-matched.", s["caption"])]
    story += [Paragraph("A4 has 2.64M trainable decoder parameters compared with S4's 4.74M. Cached decoder latency is 6.71 ms versus 7.46 ms, a 10.1% reduction. Full-pipeline latency is 209.35 ms versus 210.20 ms, a 0.4% difference. The frozen VLM therefore dominates end-to-end timing; efficiency is supporting evidence rather than the headline contribution.", s["body"])]

    story += [PageBreak(), Paragraph("8. Related work, limits, and conclusion", s["h1"])]
    story += [Paragraph("Attention maps have long been useful for grounding. Kang et al. identify a few localization-relevant heads inside a frozen full LVLM, while F-LMM uses frozen LMM attention maps with CNN and SAM refinement for referring segmentation. These establish that attention can expose grounding information, but do not compare matched trainable grounding decoders with and without FFNs. The closest broader architectural context is the controlled attention-only Transformer study in language modeling, which evaluates capacity reallocation but not natural-image grounding.", s["body"])]
    story += [Paragraph("The conclusion remains bounded. The frozen backbone retains FFNs; the output is one 24x24-derived box rather than a segmentation mask or multi-query detector; and the study does not test fine-tuning. RefCOCO and CLIP evidence is seed-0/one-seed only. A8 recovery shows a viable attention-capacity allocation, but it does not identify the mechanism of every FFN effect or show a compute-matched substitute.", s["body"])]
    story += [Paragraph("In a small grounding decoder over frozen VLM features, FFN deletion is surprisingly benign on standard and adversarial referring-expression grounding. A small controlled-compositional deficit appears at fixed depth, but attention depth recovers it and no monotonic failure boundary emerges. The supported architectural conclusion is that attention-only decoder capacity can often substitute for FFN capacity when frozen representations already supply the relevant visual and language context.", s["body"])]

    story += [PageBreak(), Paragraph("Appendix: Reproducibility Details", s["h1"])]
    story += [Paragraph("The release is evidence-first. `scripts/generate_paper_assets.py` reads committed bootstrap, slice, efficiency, CLIP-control, and RefCOCOg summary artifacts; it generates every manuscript figure, table, paper-data manifest, and website image without a GPU or model download. `scripts/verify_paper.py` regenerates these artifacts and checks the frozen tau, bootstrap seed/replicates, headline values, source hashes, expected assets, PDF text, author, and page count. `test_study.py` checks the architecture/data invariants.", s["body"])]
    reproducibility_rows = [["Artifact", "Purpose"], ["configs/study-contract.json", "Frozen model, decoder, metric, and bootstrap contract."], ["research-docs/", "Dataset card, final status, and evaluation protocol."], ["results/ and docs/results/", "Release map plus detailed aggregate evidence, bootstrap outputs, and plots."], ["decision-log/", "Compact release decisions; docs/decision-log.md preserves the chronological running record."], ["paper/data/paper-data.json", "Machine-readable source snapshot and input hashes for paper artifacts."]]
    story += [make_table(reproducibility_rows, [2.15 * inch, 5.05 * inch], s), Spacer(1, 8)]
    story += [Paragraph("The repository intentionally excludes third-party images, backbone weights, provider-local checkpoints, and raw qualitative box exports. These exclusions prevent a fabricated qualitative panel and preserve upstream license boundaries. All reported aggregate evidence, per-example metrics where versioned, slice tables, and bootstrap inputs are retained under the release record.", s["body"])]
    story += [Paragraph("AI assistance disclosure", s["h2"])]
    story += [Paragraph("AI-assisted tools were used for code assistance, experiment orchestration, figure generation, manuscript organization, and language editing. No AI system is an author; responsibility for the manuscript and any eventual submission remains with Tarun Tomar.", s["small"])]
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
