# Decision: Use FineCops-Ref and A8 to interpret the observed gap

- Date: 2026-08-14
- Status: Accepted

## Context

Ref-Adv-s did not show an A4 collapse. A controlled compositional test was needed before claiming
that FFNs are broadly dispensable. A fixed-depth gap alone could be either an FFN-specific effect
or a capacity-allocation effect.

## Decision

Evaluate frozen three-seed RefCOCOg checkpoints on FineCops-Ref and retain `A8` as the
approximately parameter-matched attention-only control. Use the locked bootstrap seed `20260812`.

## Verification

FineCops overall A4-S4 is -0.52 pp with 95% CI [-0.95, -0.12]; A8-S4 is +0.26 pp. Official level
and tuple slices are in [`docs/results/finecops/`](../docs/results/finecops/).

## Limitation

A8 is not compute-matched. The noisy official level-3 estimate cannot establish a monotonic
difficulty boundary.
