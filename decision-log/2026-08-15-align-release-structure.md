# Decision: Align the repository with the publication release standard

- Date: 2026-08-15
- Status: Accepted

## Context

The completed study had results, a review PDF, and working notes, but it did not yet expose the
same release surface as the earlier research repository: a detailed README, static research page,
canonical protocol/result indexes, citation metadata, source-package checks, and an auditable paper
structure.

## Decision

Retain the frozen experimental evidence and add a publication-facing repository layer. The release
must regenerate figures/tables from committed evidence, create a detailed review manuscript and
LaTeX source package, expose a static research page, and keep the result/decision/protocol boundary
visible without claiming new experiments.

## Verification

Run `make submission`, `python3 test_study.py`, and `make arxiv-preflight`. The paper verifier
checks result values, source hashes, release structure, generated web assets, and the rendered
manuscript.

## Limitation

The source preflight is a source-only audit until a complete TeX installation performs a clean
submission build. Human review is still required before public submission.
