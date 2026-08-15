# Decision: Freeze experiments and package a reproducible release

- Date: 2026-08-15
- Status: Accepted

## Context

The completed results support a focused architectural result. New GPU runs would not improve the
main claim without a separately justified experimental question.

## Decision

Freeze the evidence, regenerate FineCops bootstrap outputs with the locked seed, and package a
paper whose figures/tables are generated only from committed results. Keep qualitative panels out
until raw boxes and licensed images are recovered.

## Verification

`make verify-paper` regenerates the machine-readable paper evidence, figures, table, and rendered
review PDF. `python3 test_study.py` checks decoder/data invariants.

## Limitation

The release PDF is a reviewable manuscript artifact. Human review, author metadata, submission
venue selection, and any paper-hosting decision remain outside the automated release.
