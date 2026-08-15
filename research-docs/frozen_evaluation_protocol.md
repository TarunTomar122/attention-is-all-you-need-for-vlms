# Frozen Evaluation Protocol

## Question

Does removing the FFN residual from a small trainable grounding decoder materially change visual
grounding once a pretrained VLM already supplies image and text context?

## Fixed Intervention

The primary frozen SigLIP2 backbone provides 576 image tokens and contextual text states. Shared
linear projections map both streams to width 256. One learned query alternates text and image
cross-attention. `S4` adds a pre-normalized GELU FFN after each attention pair; `A4` omits it;
`A8` doubles attention-only depth. The readout, supervision, optimizer, data order, target, and
mass-to-box procedure are shared.

## Selection Boundary

The learning rate was selected by the RefCOCOg validation pilot. `tau=0.8` was selected from S4
validation once and frozen. No Ref-Adv-s or FineCops test artifact chose a checkpoint, threshold,
architecture, bin, or visualization rule.

## Metrics And Statistics

- Primary: IoU@0.5.
- Secondary: mean IoU, pointing accuracy, and target mass.
- Bootstrap: 10,000 paired image-clustered percentile replicates, seed `20260812`.
- Three-seed results: average matched seed outputs per example before resampling image IDs.
- One-seed results: descriptive only; do not supply a multi-seed confirmation claim.

## Interpretation Rules

- A4 versus S4 is the fixed-depth causal FFN deletion comparison.
- A8 addresses whether additional attention capacity can recover an observed A4 deficit; it is
  parameter-matched rather than compute-matched.
- A confidence interval crossing zero is reported as inconclusive for that slice.
- No monotonic difficulty claim is made without a consistent predeclared/official trend and
  compatible uncertainty intervals.
- The study does not rename the frozen backbone an attention-only VLM.
