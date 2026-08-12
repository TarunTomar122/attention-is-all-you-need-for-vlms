# GPU-readiness decision map

This is the canonical map of decisions that must be resolved before renting a GPU. Each answer must be supported by an artifact or a reproducible local check.

## #1: What contribution can the study defend?

Blocked by: none
Type: Research

### Question

What remains unanswered after attention-map grounding and attention-only language-model studies?

### Answer

Provisional contribution: a controlled real-image study of the **retrieval–reasoning boundary in FFN-free grounding decoders**. Existing work establishes that attention maps contain localization signal and that attention-only models can compose in synthetic worlds. It does not isolate decoder FFNs on matched real-image referring-expression tasks or characterize failures by linguistic reasoning type. See [literature-audit.md](literature-audit.md).

## #2: Which frozen backbone should carry the main claim?

Blocked by: #1
Type: Research

### Question

Which pretrained model provides sufficiently spatial patch features, usable text features, transparent licensing, and a realistic 32 GB training footprint?

### Answer

Resolved. Use SigLIP 2 B/16 at 384px as the primary backbone and OpenAI CLIP L/14 at 336px as a replication. Both expose a 24 × 24 patch grid; both remain completely frozen. See [backbone.md](backbone.md).

## #3: What exactly counts as an attention-only grounding decoder?

Blocked by: #2
Type: Discuss

### Question

How do both variants produce the same localization output while differing only in the presence of decoder FFNs?

### Answer

Resolved. A single grounding query alternates text and image cross-attention across four blocks. The standard variant adds a 4× GELU FFN to every block; the attention-only variant deletes those FFNs. A separate shared Q/K attention readout converts the final query into the only localization prediction: a distribution over 576 patches. See [architecture.md](architecture.md).

## #4: Which controls make the FFN comparison fair?

Blocked by: #3
Type: Discuss

### Question

Which depth-, parameter-, and compute-matched comparisons are necessary to separate architecture from capacity?

### Answer

Unresolved.

## #5: Which datasets answer which part of the question?

Blocked by: #1
Type: Research

### Question

Which official splits support training, shortcut-resistant evaluation, long expressions, hard distractors, and reasoning-facet analysis?

### Answer

Unresolved. Current candidates: RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv. Dataset availability and annotation schemas must be verified before locking them.

## #6: How are task types assigned before results are seen?

Blocked by: #5
Type: Prototype

### Question

Can deterministic, auditable rules assign category, attribute, absolute-position, relation, comparison, ordinal, counting, negation, and compositional tags without turning noisy heuristics into ground truth?

### Answer

Unresolved. Prefer native benchmark facets where available; use multi-label lexical rules plus a fixed manual audit for legacy datasets.

## #7: What evidence will support or reject the hypothesis?

Blocked by: #3, #4, #6
Type: Discuss

### Question

Which metrics, seeds, intervals, ablations, faithfulness tests, and success thresholds must be fixed before training?

### Answer

Unresolved.

## #8: What must work locally before GPU rental?

Blocked by: #2, #3, #5, #6, #7
Type: Prototype

### Question

What is the smallest CPU-tested implementation and exact runbook that turns GPU time directly into a smoke run and then experiments?

### Answer

Unresolved.
