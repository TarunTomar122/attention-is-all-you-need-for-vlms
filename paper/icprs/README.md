# ICPRS 2027 submission package

This directory is a double-blind IEEE-style adaptation of the canonical paper.
The arXiv manuscript in `../main.tex` remains unchanged.

## Status

- Target: ICPRS 2027, Regular Paper or Regular Student Paper.
- Deadline: 25 October 2026, 23:59 UK time.
- Format target: 6 pages plus up to 1 page of references.
- Authors: anonymized for double-blind review.
- GPU reruns: not required.

## Build

From this directory:

```bash
cp ../references.bib references.bib
tectonic --keep-logs --keep-intermediates main.tex
pdfinfo main.pdf | rg 'Pages|Author|Title'
```

The source uses the generated vector figures from `../figures/`. Before submission, verify that
the compiled body is at most six pages and references occupy at most one additional page.

## Submission choices

Use **Regular Student Paper** only if the first author is officially registered as a student on
the submission date. Otherwise use **Regular Paper**. At least one author must later register at
the full author fee for publication and presentation.

## Blind-review checklist

- [x] Author name, affiliation, GitHub URL, acknowledgements, and local paths removed.
- [x] PDF metadata author set to `Anonymous`.
- [x] No self-citation or public repository link in the manuscript.
- [ ] Confirm page count after the final IEEE template compilation.
- [ ] Confirm all authors and student status in ConfTool.
- [ ] Add any required camera-ready acknowledgements after review.
