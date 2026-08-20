# Paper Package

Title: **Do Visual Grounding Decoders Need Feed-Forward Networks?**

This directory contains the canonical manuscript source and the CPU-only path from committed
experiment evidence to paper figures, tables, a reviewable PDF, and a submission-source archive.
No model weights, dataset images, or GPU are required.

## Start Here

1. Read the rendered [`main.pdf`](main.pdf) and editable [`main.tex`](main.tex).
2. Check each scientific sentence against [`claims-and-limitations.md`](claims-and-limitations.md).
3. Use [`method.md`](method.md) and the [frozen protocol](../research-docs/frozen_evaluation_protocol.md)
   to audit the intervention and statistics.
4. Read [`citation-audit.md`](citation-audit.md) before changing related-work wording.
5. Use [`submission-checklist.md`](submission-checklist.md) only after the human author has reviewed
   the manuscript.

## Regenerate Everything

```bash
brew install tectonic
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

The command creates PNG/PDF/SVG figure variants, the CSV/Markdown/LaTeX result table, the
machine-readable paper-data manifest, static-site figures, and the rendered review PDF. It checks
the frozen contract and exact headline result values.

## Build Or Package The Source

```bash
make paper-pdf
make overleaf-package
make arxiv-package
make arxiv-preflight
make icprs-pdf
make icprs-package
```

`paper-pdf` compiles the canonical LaTeX manuscript with Tectonic. The Overleaf
and arXiv ZIPs contain `main.tex`, `references.bib`, the generated LaTeX table, and all referenced
vector PDF figures. `arxiv-preflight` performs source hygiene, citation, and inclusion checks.

## ICPRS adaptation

[`icprs/`](icprs/) contains the anonymous IEEEtran conference-format adaptation for ICPRS: a
compact paper within the six-page paper plus one-page reference limit, with identifying metadata
and public repository links removed. It currently renders as seven pages total (six body pages and
one references page). Its source archive is built by `make icprs-package`. Keep the
canonical manuscript above unchanged; choose **Regular Student Paper** in ConfTool only if the
first author is officially registered as a student by the submission deadline, otherwise choose
**Regular Paper**.

## What Is Canonical

- Keep: `main.tex`, `main.pdf`, `references.bib`, `figures/*.pdf`, the generated table, evidence
  manifest, and research notes.
- Generated and ignored: `overleaf-package.zip`, `arxiv-source.zip`, and LaTeX intermediates.
- Website PNGs are generated once under `../docs/assets/`.
- `draft.md`, `outline.md`, and `writing-guide.md` are writing/review aids, not evidence sources.

## Figure Map

| Figure | Main point |
| --- | --- |
| `generated-evidence-overview` | The completed fixed-depth comparisons and intervals. |
| `generated-method-overview` | The FFN residual is the only A4/S4 intervention. |
| `generated-refadv-performance` | Absolute adversarial performance by native difficulty metadata. |
| `generated-refadv-deltas` | Ref-Adv does not expose a monotonic A4 deficit. |
| `generated-finecops-performance` | All variants decline with official FineCops difficulty. |
| `generated-finecops-deltas` | FineCops slice gaps are not monotonic. |
| `generated-finecops-difficulty` | FineCops has a small overall gap without monotonic difficulty evidence. |
| `generated-efficiency-summary` | A4 reduces decoder parameters/latency; end-to-end timing stays backbone-dominated. |

## Evidence Boundary

The paper is about an FFN-free trainable grounding decoder over frozen VLM features. It does not
claim an attention-only VLM, pixel segmentation, multiple-object grounding, backbone fine-tuning,
or universal FFN dispensability. See [`data-availability.md`](data-availability.md) for the raw
asset and license boundary.
