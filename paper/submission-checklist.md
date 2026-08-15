# Submission checklist

## Completed scientific freeze

- [x] FineCops bootstrap/plots use locked seed `20260812`.
- [x] No model rerun, threshold change, or post-hoc dataset tuning after the freeze.
- [x] Generated table/figures read only committed result artifacts.
- [x] Claim boundaries distinguish confirmatory, descriptive, and noisy evidence.

## Before public submission

- [x] Add author and affiliation metadata to the release draft.
- [x] Complete citation audit against primary sources.
- [x] Render and visually inspect the release-draft PDF.
- [ ] Human-review every number against `paper/data/paper-data.json`.
- [x] Add data/checkpoint availability and redistribution boundary.

## Optional enhancement, not a release blocker

- [ ] Add 4–6 qualitative examples only if raw target/predicted box exports and licensed images are restored; do not reconstruct them from aggregate tables.
- [ ] Create an anonymized copy if the venue requires it.

## Release boundary

Publish code, manifests, scripts, per-example metrics, bootstrap inputs, and plots. Keep raw COCO/GQA/Ref-Adv images and provider-local checkpoints out of Git unless their licenses and hosting terms permit redistribution.
