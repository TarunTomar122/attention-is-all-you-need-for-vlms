# Decision log

Append dated decisions here; include the evidence and what would change the decision.

## 2026-08-12 — Start with frozen-VLM referring-expression grounding

- **Decision:** Compare a standard grounding decoder with an FFN-free attention-only decoder over frozen VLM features.
- **Why:** Continuous robot action generation overloads an attention-only expert. Referring-expression grounding directly tests contextual selection over real images.
- **Initial benchmark:** RefCOCO; report accuracy at IoU >= 0.5 and break out relational expressions.
- **Revisit when:** A literature review finds an identical frozen-VLM, real-image, matched-decoder ablation.
