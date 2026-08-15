# Paper package

Canonical manuscript source: [`main.tex`](main.tex). The rendered review draft is [`main.pdf`](main.pdf), and the readable working draft is [`draft.md`](draft.md).

The package follows the same evidence-first workflow as the prior `vision-pathways` paper: every figure and table is generated from committed result artifacts, then checked without a GPU.

```bash
make paper-assets
make verify-paper
make paper-pdf  # requires a local LaTeX installation
```

## Paper story

**Do Visual Grounding Decoders Need Feed-Forward Networks?** studies one small trainable grounding decoder over frozen VLM image/text tokens. At the matched fixed depth, A4 removes the FFN from each of four decoder blocks; S4 keeps it. A8 uses eight attention-only blocks to approximately match S4's parameter count.

The evidence supports a narrow conclusion: FFNs are often dispensable in this decoder, and where a small controlled-compositional gap appears, additional attention depth recovers it. It does **not** establish an attention-only VLM or universal FFN dispensability.

- [`outline.md`](outline.md): final section-level argument.
- [`claims-and-limitations.md`](claims-and-limitations.md): sentence-level claim boundaries.
- [`method.md`](method.md): exact architecture and evaluation contract.
- [`related-work.md`](related-work.md): novelty positioning.
- [`citation-audit.md`](citation-audit.md): primary-source bibliography verification.
- [`data-availability.md`](data-availability.md): versioned artifacts and license boundary.
- [`submission-checklist.md`](submission-checklist.md): remaining human-facing work.
- [`data/paper-data.json`](data/paper-data.json): generated evidence snapshot and hashes.
