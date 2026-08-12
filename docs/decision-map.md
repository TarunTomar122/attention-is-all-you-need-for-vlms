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

Resolved. `S4` versus `A4` is the same-depth causal ablation and is already approximately compute-matched because FFNs process only one query. `A8` reallocates the removed FFN parameters into attention depth and matches `S4` trainable parameters within about 0.2%. Depth sweeps and non-trained modality controls complete the matrix. See [controls.md](controls.md).

## #5: Which datasets answer which part of the question?

Blocked by: #1
Type: Research

### Question

Which official splits support training, shortcut-resistant evaluation, long expressions, hard distractors, and reasoning-facet analysis?

### Answer

Resolved. RefCOCOg UMD carries the full study; RefCOCO+ UNC and RefCOCO UNC carry the core replication; each is trained separately. Ref-Adv-s is a locked, test-only OOD benchmark for RefCOCOg-trained models. Official schemas, expression counts, archived annotation checksums, split isolation, and loader invariants are recorded in [datasets.md](datasets.md).

## #6: How are task types assigned before results are seen?

Blocked by: #5
Type: Prototype

### Question

Can deterministic, auditable rules assign category, attribute, absolute-position, relation, comparison, ordinal, counting, negation, and compositional tags without turning noisy heuristics into ground truth?

### Answer

Resolved. Use auditable multi-label lexical tags plus mutually exclusive direct, absolute, relational, logical, and unclassified strata. The abstention bucket prevents long unmatched relations from contaminating the confirmatory direct slice. A compositional overlay captures multiple structural cues; expression length remains an independent covariate. Audit 40 RefCOCOg training examples per stratum, revise only from that audit, then freeze before validation, test, or Ref-Adv-s is loaded. Native Ref-Adv-s fields remain separate diagnostics. See [task-taxonomy.md](task-taxonomy.md).

## #7: What evidence will support or reject the hypothesis?

Blocked by: #3, #4, #6
Type: Discuss

### Question

Which metrics, seeds, intervals, ablations, faithfulness tests, and success thresholds must be fixed before training?

### Answer

Resolved. The confirmatory RefCOCOg comparison requires `A4` to remain within a 5-point practical margin of `S4` on direct references and a task interaction showing at least a 5-point larger deficit on logical references. Use image-clustered paired bootstrap intervals, fixed interpretation gates, heatmap diagnostics, per-seed reporting, and descriptive replication analyses. See [evaluation.md](evaluation.md).

## #8: What must work locally before GPU rental?

Blocked by: #2, #3, #5, #6, #7
Type: Prototype

### Question

What is the smallest CPU-tested implementation and exact runbook that turns GPU time directly into a smoke run and then experiments?

### Answer

Resolved. The repository now contains deterministic preparation, decoder, training, evaluation, threshold-selection, and paired-analysis entrypoints plus one focused CPU invariant check. The [GPU runbook](gpu-runbook.md) begins with exact environment/data preparation, then a bounded CUDA smoke run, the shared learning-rate pilot, primary matrix, frozen threshold selection, test export, and confirmatory analysis.
