# Decision Log

This directory is the permanent, human-readable account of why the study changed. Git records
what changed; these entries record the evidence, constraints, and remaining uncertainty behind the
research decisions.

## Entry Format

Each release-facing decision has one dated Markdown entry:

```text
YYYY-MM-DD-short-description.md
```

Each entry records context, decision, changes, verification, limitations, and the next enabled
step. A decision entry is not evidence that an experiment succeeded; completed claims still require
saved predictions, frozen configurations, and uncertainty analysis.

The historical running notes remain in [`docs/decision-log.md`](../docs/decision-log.md). These
entries are the compact canonical release record.
