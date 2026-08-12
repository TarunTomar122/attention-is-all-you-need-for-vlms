# Attention Is All You Need for VLMs

Can an attention-only grounding decoder localize a language-referred object from frozen pretrained VLM features?

Initial study: characterize where an FFN-free grounding decoder succeeds or fails. Given an image and expression, predict one bounding box; compare matched standard and attention-only decoders across retrieval-style and compositional references.

Project records live in [`docs/`](docs/).
